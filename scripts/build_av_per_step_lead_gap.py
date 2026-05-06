#!/usr/bin/env python3
"""Build per-step ego-to-lead gap cache from Waymo actor_tracks.

For each (scenario_id, step_index) in the ego replay surface, identifies the
closest same-lane-corridor lead vehicle and records the inter-vehicle gap.

Lead identification uses the **same-lane corridor** heuristic:
    - Transform other actors into the ego's local (body) frame.
    - Filter to actors ahead (rel_x > 0) and laterally close (|rel_y| < 1.75 m).
    - Select the nearest qualifying actor (minimum rel_x) as the lead.

Output: data/orius_av/av/processed_full_corpus/per_step_lead_gap.parquet
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Half US lane width (3.5 m / 2) — corridor filter threshold
_HALF_LANE_M = 1.75

_BASE = Path("data/orius_av/av/processed_full_corpus")
_ACTOR_TRACKS = _BASE / "actor_tracks.parquet"
_REPLAY_WINDOWS = _BASE / "replay_windows.parquet"
_OUTPUT = _BASE / "per_step_lead_gap.parquet"


def _load_ego_positions(actor_tracks: pd.DataFrame) -> pd.DataFrame:
    """Extract ego rows with position, speed, heading per (scenario, step)."""
    ego = actor_tracks.loc[
        actor_tracks["is_ego"] & actor_tracks["valid"],
        ["scenario_id", "step_index", "x_m", "y_m", "speed_mps", "heading_rad"],
    ].copy()
    ego.rename(
        columns={
            "x_m": "ego_x_m",
            "y_m": "ego_y_m",
            "speed_mps": "ego_speed_mps",
            "heading_rad": "ego_heading_rad",
        },
        inplace=True,
    )
    return ego


def _compute_lead_gap(actor_tracks: pd.DataFrame, ego: pd.DataFrame) -> pd.DataFrame:
    """For every ego (scenario, step), find the closest same-lane lead."""
    # Get non-ego valid actors
    others = actor_tracks.loc[
        ~actor_tracks["is_ego"] & actor_tracks["valid"],
        ["scenario_id", "step_index", "track_id", "x_m", "y_m", "speed_mps", "heading_rad"],
    ].copy()

    # Merge ego position onto other actors
    merged = others.merge(ego, on=["scenario_id", "step_index"], how="inner")

    # Transform to ego-local frame
    dx = merged["x_m"].values - merged["ego_x_m"].values
    dy = merged["y_m"].values - merged["ego_y_m"].values
    h = merged["ego_heading_rad"].values
    cos_h = np.cos(h)
    sin_h = np.sin(h)
    merged["rel_x"] = dx * cos_h + dy * sin_h
    merged["rel_y"] = -dx * sin_h + dy * cos_h

    # Same-lane corridor filter: ahead and within half-lane width
    mask = (merged["rel_x"] > 0) & (merged["rel_y"].abs() < _HALF_LANE_M)
    candidates = merged.loc[mask].copy()

    # Select closest lead per (scenario, step)
    if len(candidates) == 0:
        lead = pd.DataFrame(
            columns=[
                "scenario_id",
                "step_index",
                "lead_track_id",
                "lead_rel_x_m",
                "lead_rel_y_m",
                "lead_speed_mps",
                "lead_heading_rad",
            ],
        )
    else:
        idx = candidates.groupby(["scenario_id", "step_index"])["rel_x"].idxmin()
        lead = candidates.loc[
            idx,
            [
                "scenario_id",
                "step_index",
                "track_id",
                "rel_x",
                "rel_y",
                "speed_mps",
                "heading_rad",
            ],
        ].rename(
            columns={
                "track_id": "lead_track_id",
                "rel_x": "lead_rel_x_m",
                "rel_y": "lead_rel_y_m",
                "speed_mps": "lead_speed_mps",
                "heading_rad": "lead_heading_rad",
            }
        )

    return lead


def build_per_step_lead_gap(
    actor_tracks_path: Path = _ACTOR_TRACKS,
    replay_windows_path: Path = _REPLAY_WINDOWS,
    output_path: Path = _OUTPUT,
) -> pd.DataFrame:
    """Build and save the per-step lead-gap parquet."""
    print(f"Loading actor_tracks from {actor_tracks_path} ...")
    t0 = time.time()
    at = pd.read_parquet(actor_tracks_path)
    print(f"  Loaded {len(at):,} rows in {time.time() - t0:.1f}s")

    print("Extracting ego positions ...")
    ego = _load_ego_positions(at)
    print(f"  {len(ego):,} ego (scenario, step) rows")

    print("Computing per-step lead gaps ...")
    t1 = time.time()
    lead = _compute_lead_gap(at, ego)
    print(f"  Found {len(lead):,} steps with a same-lane lead in {time.time() - t1:.1f}s")

    # Left-join ego rows with lead info
    result = ego.merge(lead, on=["scenario_id", "step_index"], how="left")
    result["lead_present"] = result["lead_track_id"].notna()

    # Add ts_utc from replay_windows if available
    rw = pd.read_parquet(replay_windows_path, columns=["scenario_id", "step_index", "ts_utc"])
    result = result.merge(rw, on=["scenario_id", "step_index"], how="left")

    # Reorder columns
    col_order = [
        "scenario_id",
        "step_index",
        "ts_utc",
        "ego_x_m",
        "ego_y_m",
        "ego_speed_mps",
        "ego_heading_rad",
        "lead_present",
        "lead_track_id",
        "lead_rel_x_m",
        "lead_rel_y_m",
        "lead_speed_mps",
        "lead_heading_rad",
    ]
    for c in col_order:
        if c not in result.columns:
            result[c] = np.nan
    result = result[col_order]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False)
    print(f"Wrote {len(result):,} rows to {output_path}")

    n_with_lead = result["lead_present"].sum()
    print(f"  Lead present in {n_with_lead:,}/{len(result):,} steps ({100 * n_with_lead / len(result):.1f}%)")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor-tracks", type=Path, default=_ACTOR_TRACKS)
    parser.add_argument("--replay-windows", type=Path, default=_REPLAY_WINDOWS)
    parser.add_argument("-o", "--output", type=Path, default=_OUTPUT)
    args = parser.parse_args()
    build_per_step_lead_gap(args.actor_tracks, args.replay_windows, args.output)
