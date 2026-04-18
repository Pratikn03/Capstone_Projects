#!/usr/bin/env python3
"""Augment AV trajectories with RSS collision-gap safety predicate (Path B).

Reads the Path A bridge outputs (av_trajectories_orius.csv, features.parquet)
and the per-step lead-gap cache, then adds RSS columns:
  - lead_present, lead_rel_x_m, lead_speed_mps
  - rss_safe_gap_m, rss_violation_true

Backward compatibility: all original HEE columns are preserved.
New columns are additive (optional-read by downstream loaders).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = REPO_ROOT / "data" / "orius_av" / "av" / "processed"
FULL_CORPUS = REPO_ROOT / "data" / "orius_av" / "av" / "processed_full_corpus"
LEAD_GAP = FULL_CORPUS / "per_step_lead_gap.parquet"


def _parse_vehicle_id(vehicle_id: str) -> tuple[str, int]:
    """Split 'scenarioId_trackId' back into components."""
    parts = vehicle_id.rsplit("_", 1)
    return parts[0], int(parts[1])


def augment_with_rss(
    trajectories_path: Path = PROCESSED / "av_trajectories_orius.csv",
    features_path: Path = PROCESSED / "features.parquet",
    lead_gap_path: Path = LEAD_GAP,
    output_csv: Path = PROCESSED / "av_trajectories_orius.csv",
    output_features: Path = PROCESSED / "features.parquet",
    t_resp: float = 0.75,
    a_min_brake_ego: float = 4.0,
    a_max_brake_lead: float = 6.0,
) -> dict[str, int]:
    """Augment trajectory CSV and features parquet with RSS columns."""
    # Lazy import to keep script self-contained
    import importlib
    mod = importlib.import_module("orius.vehicles.rss_safety")
    rss_safe_gap_vec = mod.rss_safe_gap_vec
    RssParameters = mod.RssParameters

    params = RssParameters(t_resp=t_resp, a_min_brake_ego=a_min_brake_ego,
                           a_max_brake_lead=a_max_brake_lead)

    # Load lead-gap cache
    gap_df = pd.read_parquet(lead_gap_path)
    print(f"[rss-bridge] lead-gap cache: {len(gap_df):,} rows, "
          f"{gap_df.lead_present.sum():,} with lead")

    # Load trajectories
    traj = pd.read_csv(trajectories_path)
    print(f"[rss-bridge] trajectories: {len(traj):,} rows")

    # Parse vehicle_id → scenario_id + ego_track_id to join with lead-gap
    parsed = traj["vehicle_id"].apply(_parse_vehicle_id)
    traj["scenario_id"] = [p[0] for p in parsed]
    traj["step_index"] = traj["step"]

    # Merge lead-gap onto trajectories
    lead_cols = ["scenario_id", "step_index", "lead_present", "lead_track_id",
                 "lead_rel_x_m", "lead_rel_y_m", "lead_speed_mps"]
    traj = traj.merge(
        gap_df[lead_cols],
        on=["scenario_id", "step_index"],
        how="left",
    )
    traj["lead_present"] = traj["lead_present"].fillna(False)

    # Compute RSS safe gap and violation
    v_ego = traj["speed_mps"].values.astype(np.float64)
    v_lead = traj["lead_speed_mps"].fillna(0.0).values.astype(np.float64)
    safe_gap = rss_safe_gap_vec(v_ego, v_lead, params=params)
    traj["rss_safe_gap_m"] = safe_gap

    # Violation: only meaningful when lead is present
    traj["rss_violation_true"] = False
    mask = traj["lead_present"].values
    actual_gap = traj["lead_rel_x_m"].values
    traj.loc[mask, "rss_violation_true"] = actual_gap[mask] < safe_gap[mask]

    # Update lead_position_m to reflect real lead (where present)
    # This replaces the placeholder +500m from Path A
    traj.loc[mask, "lead_position_m"] = (
        traj.loc[mask, "position_m"].values + traj.loc[mask, "lead_rel_x_m"].values
    )

    # Write augmented CSV (drop internal join columns)
    csv_cols = [
        "vehicle_id", "step", "position_m", "speed_mps", "speed_limit_mps",
        "lead_position_m", "ts_utc",
        "lead_present", "lead_rel_x_m", "lead_speed_mps",
        "rss_safe_gap_m", "rss_violation_true",
    ]
    traj[csv_cols].to_csv(output_csv, index=False)
    print(f"[rss-bridge] wrote {len(traj):,} rows to {output_csv}")

    # Augment features.parquet
    feat = pd.read_parquet(features_path)
    print(f"[rss-bridge] features: {len(feat):,} rows")

    # Parse vehicle_id in features too
    feat_parsed = feat["vehicle_id"].apply(_parse_vehicle_id)
    feat["scenario_id"] = [p[0] for p in feat_parsed]
    feat["step_index"] = feat["step"]

    feat = feat.merge(
        gap_df[lead_cols],
        on=["scenario_id", "step_index"],
        how="left",
    )
    feat["lead_present"] = feat["lead_present"].fillna(False)

    v_ego_f = feat["speed_mps"].values.astype(np.float64)
    v_lead_f = feat["lead_speed_mps"].fillna(0.0).values.astype(np.float64)
    safe_gap_f = rss_safe_gap_vec(v_ego_f, v_lead_f, params=params)
    feat["rss_safe_gap_m"] = safe_gap_f
    feat["rss_violation_true"] = False
    mask_f = feat["lead_present"].values
    actual_gap_f = feat["lead_rel_x_m"].values
    feat.loc[mask_f, "rss_violation_true"] = actual_gap_f[mask_f] < safe_gap_f[mask_f]

    # Update lead_position_m
    feat.loc[mask_f, "lead_position_m"] = (
        feat.loc[mask_f, "position_m"].values + feat.loc[mask_f, "lead_rel_x_m"].values
    )

    # Drop join columns, keep schema additive
    drop_cols = ["scenario_id", "step_index", "lead_track_id", "lead_rel_y_m"]
    feat_out = feat.drop(columns=[c for c in drop_cols if c in feat.columns])
    feat_out.to_parquet(output_features, index=False)
    print(f"[rss-bridge] wrote {len(feat_out):,} features to {output_features}")

    # Stats
    n_viol_traj = traj["rss_violation_true"].sum()
    n_viol_feat = feat["rss_violation_true"].sum()
    stats = {
        "trajectory_rows": len(traj),
        "feature_rows": len(feat_out),
        "lead_present_traj": int(traj["lead_present"].sum()),
        "rss_violations_traj": int(n_viol_traj),
        "rss_violation_rate_traj": float(n_viol_traj / max(1, traj["lead_present"].sum())),
        "rss_violations_feat": int(n_viol_feat),
    }
    print(f"[rss-bridge] RSS violations (trajectories): {n_viol_traj:,} / "
          f"{int(traj['lead_present'].sum()):,} lead-present steps "
          f"({stats['rss_violation_rate_traj']:.2%})")
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectories", type=Path,
                        default=PROCESSED / "av_trajectories_orius.csv")
    parser.add_argument("--features", type=Path,
                        default=PROCESSED / "features.parquet")
    parser.add_argument("--lead-gap", type=Path, default=LEAD_GAP)
    parser.add_argument("--t-resp", type=float, default=0.75)
    parser.add_argument("--a-min-brake-ego", type=float, default=4.0)
    parser.add_argument("--a-max-brake-lead", type=float, default=6.0)
    args = parser.parse_args()

    augment_with_rss(
        trajectories_path=args.trajectories,
        features_path=args.features,
        lead_gap_path=args.lead_gap,
        t_resp=args.t_resp,
        a_min_brake_ego=args.a_min_brake_ego,
        a_max_brake_lead=args.a_max_brake_lead,
    )
