#!/usr/bin/env python3
"""Translate Waymo full-corpus AV surfaces into the HEE legacy schema.

The canonical AV registry entry and downstream consumers
(``real_data_loader.load_vehicle_rows``, legacy training config,
``run_all_domain_eval.py``) expect the HEE 1D kinematic schema:

    vehicle_id, step, position_m, speed_mps, speed_limit_mps,
    lead_position_m, ts_utc, timestamp

The Waymo full-corpus pipeline (``orius.av_waymo``) produces a scenario-native
2D schema under ``data/orius_av/av/processed_full_corpus/``.  This bridge
reads the Waymo replay windows plus scenario-level splits and emits HEE-schema
parquets + runtime CSV that replace the current 85-row HEE legacy surface,
giving the canonical path ~1,300 training scenarios (~120k rows) without
changing any downstream consumer.

Path A semantics: ego-only 1D speed-limit track.  lead_position_m is pinned
at a far constant so it is never the binding safety predicate; Path B will
replace it with neighbor-anchor-derived collision geometry.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from orius.vehicles.rss_safety import RssParameters, rss_safe_gap_vec

REPO_ROOT = Path(__file__).resolve().parents[1]
FULL_CORPUS = REPO_ROOT / "data" / "orius_av" / "av" / "processed_full_corpus"
PROCESSED = REPO_ROOT / "data" / "orius_av" / "av" / "processed"

SPEED_LIMIT_DEFAULT_MPS = 30.0
LEAD_GAP_CONSTANT_M = 500.0
LAG_STEPS = (1, 2, 4, 8, 12, 24)
LEAD_GAP_PATH = FULL_CORPUS / "per_step_lead_gap.parquet"


def _load_replay_windows(full_corpus: Path) -> pd.DataFrame:
    df = pd.read_parquet(full_corpus / "replay_windows.parquet")
    required = {
        "scenario_id",
        "step_index",
        "ego_track_id",
        "ego_x_m",
        "ego_y_m",
        "ego_speed_mps",
        "timestamp_us",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"replay_windows.parquet missing columns: {sorted(missing)}")
    return df.sort_values(["scenario_id", "ego_track_id", "step_index"]).reset_index(drop=True)


def _load_split_scenarios(full_corpus: Path, name: str) -> set[str]:
    path = full_corpus / "splits" / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"expected split file missing: {path}")
    scenarios = pd.read_parquet(path)["scenario_id"].astype(str).unique().tolist()
    return set(scenarios)


def _cumulative_arc_length(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    if len(x) == 0:
        return np.zeros(0)
    dx = np.diff(x, prepend=x[0])
    dy = np.diff(y, prepend=y[0])
    seg = np.sqrt(dx * dx + dy * dy)
    return np.cumsum(seg)


def _translate(df: pd.DataFrame) -> pd.DataFrame:
    out_frames: list[pd.DataFrame] = []
    for (scenario_id, ego_track_id), grp in df.groupby(["scenario_id", "ego_track_id"], sort=False):
        g = grp.reset_index(drop=True)
        position_m = _cumulative_arc_length(g["ego_x_m"].to_numpy(), g["ego_y_m"].to_numpy())
        vehicle_id = f"{scenario_id}_{ego_track_id}"
        frame = pd.DataFrame(
            {
                "vehicle_id": vehicle_id,
                "step": g["step_index"].astype(int).to_numpy(),
                "position_m": position_m,
                "speed_mps": g["ego_speed_mps"].astype(float).to_numpy(),
                "speed_limit_mps": SPEED_LIMIT_DEFAULT_MPS,
                "lead_position_m": position_m + LEAD_GAP_CONSTANT_M,
                "timestamp_us": g["timestamp_us"].astype("int64").to_numpy(),
                "scenario_id": scenario_id,
            }
        )
        frame["ts_utc"] = pd.to_datetime(frame["timestamp_us"], unit="us", utc=True).dt.strftime(
            "%Y-%m-%dT%H:%M:%S.%fZ"
        )
        frame["timestamp"] = pd.to_datetime(frame["timestamp_us"], unit="us", utc=True)
        out_frames.append(frame)
    if not out_frames:
        raise ValueError("no scenarios produced; replay_windows empty after groupby")
    return pd.concat(out_frames, ignore_index=True)


def _augment_with_rss(
    trajectories_df: pd.DataFrame,
    lead_gap_path: Path,
    *,
    params: RssParameters | None = None,
) -> pd.DataFrame:
    if not lead_gap_path.exists():
        print(f"[bridge] lead-gap cache missing at {lead_gap_path}; leaving Path A gap semantics")
        return trajectories_df

    gap_df = pd.read_parquet(lead_gap_path)
    required = {
        "scenario_id",
        "step_index",
        "lead_present",
        "lead_track_id",
        "lead_rel_x_m",
        "lead_rel_y_m",
        "lead_speed_mps",
    }
    missing = required - set(gap_df.columns)
    if missing:
        print(f"[bridge] lead-gap cache missing columns {sorted(missing)}; leaving Path A gap semantics")
        return trajectories_df

    lead_cols = [
        "scenario_id",
        "step_index",
        "lead_present",
        "lead_track_id",
        "lead_rel_x_m",
        "lead_rel_y_m",
        "lead_speed_mps",
    ]
    out = trajectories_df.merge(
        gap_df[lead_cols],
        left_on=["scenario_id", "step"],
        right_on=["scenario_id", "step_index"],
        how="left",
    )
    out["lead_present"] = out["lead_present"].fillna(False).astype(bool)

    v_ego = out["speed_mps"].astype(float).to_numpy()
    v_lead = out["lead_speed_mps"].fillna(0.0).astype(float).to_numpy()
    safe_gap = rss_safe_gap_vec(v_ego, v_lead, params=params or RssParameters())
    out["rss_safe_gap_m"] = safe_gap
    out["rss_violation_true"] = False

    actual_gap = out["lead_rel_x_m"].fillna(LEAD_GAP_CONSTANT_M).astype(float).to_numpy()
    mask = out["lead_present"].to_numpy(dtype=bool)
    out.loc[mask, "rss_violation_true"] = actual_gap[mask] < safe_gap[mask]
    out["lead_position_m"] = out["position_m"] + LEAD_GAP_CONSTANT_M
    out.loc[mask, "lead_position_m"] = out.loc[mask, "position_m"].to_numpy() + actual_gap[mask]

    print(
        "[bridge] rss augmentation: "
        f"lead-present={int(out['lead_present'].sum()):,}, "
        f"violations={int(out['rss_violation_true'].sum()):,}"
    )
    return out.drop(columns=["step_index"], errors="ignore")


def _add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["vehicle_id", "step"]).reset_index(drop=True)
    for lag in LAG_STEPS:
        out[f"speed_mps_lag{lag}"] = out.groupby("vehicle_id", sort=False)["speed_mps"].shift(lag)
        out[f"position_m_lag{lag}"] = out.groupby("vehicle_id", sort=False)["position_m"].shift(lag)
    out["hour"] = out["timestamp"].dt.hour
    out["minute"] = out["timestamp"].dt.minute
    return out.dropna(subset=[f"speed_mps_lag{max(LAG_STEPS)}"]).reset_index(drop=True)


def _filter_to_scenarios(df: pd.DataFrame, scenarios: set[str]) -> pd.DataFrame:
    return df[df["scenario_id"].isin(scenarios)].reset_index(drop=True)


def _backup_legacy(processed: Path) -> Path | None:
    legacy_markers = ["features.parquet", "av_trajectories_orius.csv", "splits"]
    to_move = [processed / m for m in legacy_markers if (processed / m).exists()]
    if not to_move:
        return None
    backup_dir = processed / "hee_legacy"
    if backup_dir.exists():
        return backup_dir
    backup_dir.mkdir(parents=True, exist_ok=True)
    for path in to_move:
        shutil.move(str(path), str(backup_dir / path.name))
    return backup_dir


def _write_outputs(
    features_df: pd.DataFrame,
    trajectories_df: pd.DataFrame,
    scenarios_by_split: dict[str, set[str]],
    processed: Path,
) -> dict[str, int]:
    processed.mkdir(parents=True, exist_ok=True)
    splits_dir = processed / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    feature_cols_drop = ["scenario_id", "timestamp_us"]
    features_out = features_df.drop(columns=[c for c in feature_cols_drop if c in features_df.columns])
    features_out.to_parquet(processed / "features.parquet", index=False)

    split_row_counts: dict[str, int] = {"features": len(features_out)}
    for split_name, scenarios in scenarios_by_split.items():
        sub = _filter_to_scenarios(features_df, scenarios).drop(
            columns=[c for c in feature_cols_drop if c in features_df.columns]
        )
        out_path = splits_dir / f"{split_name}.parquet"
        sub.to_parquet(out_path, index=False)
        split_row_counts[split_name] = len(sub)

    trajectory_cols = [
        "vehicle_id",
        "step",
        "position_m",
        "speed_mps",
        "speed_limit_mps",
        "lead_position_m",
        "ts_utc",
    ]
    optional_trajectory_cols = [
        "lead_present",
        "lead_track_id",
        "lead_rel_x_m",
        "lead_rel_y_m",
        "lead_speed_mps",
        "rss_safe_gap_m",
        "rss_violation_true",
    ]
    trajectory_cols.extend([col for col in optional_trajectory_cols if col in trajectories_df.columns])
    trajectories_df[trajectory_cols].to_csv(processed / "av_trajectories_orius.csv", index=False)
    split_row_counts["av_trajectories_orius_csv"] = len(trajectories_df)
    return split_row_counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-corpus", type=Path, default=FULL_CORPUS)
    parser.add_argument("--processed", type=Path, default=PROCESSED)
    parser.add_argument("--no-backup", action="store_true", help="Skip backing up HEE legacy files")
    parser.add_argument("--dry-run", action="store_true", help="Print counts without writing")
    args = parser.parse_args()

    print(f"[bridge] reading Waymo full corpus from {args.full_corpus}")
    replay = _load_replay_windows(args.full_corpus)
    print(f"[bridge] replay_windows: {len(replay):,} rows, {replay['scenario_id'].nunique()} scenarios")

    scenarios_by_split = {
        name: _load_split_scenarios(args.full_corpus, name)
        for name in ("train", "calibration", "val", "test")
    }
    all_split_scenarios = set().union(*scenarios_by_split.values())
    print(
        f"[bridge] split scenarios: train={len(scenarios_by_split['train'])} cal={len(scenarios_by_split['calibration'])} val={len(scenarios_by_split['val'])} test={len(scenarios_by_split['test'])}"
    )

    trajectories_df = _translate(replay)
    print(
        f"[bridge] translated trajectories: {len(trajectories_df):,} rows, {trajectories_df['vehicle_id'].nunique()} vehicles"
    )
    trajectories_df = _augment_with_rss(trajectories_df, args.full_corpus / LEAD_GAP_PATH.name)

    trajectories_in_splits = _filter_to_scenarios(trajectories_df, all_split_scenarios)
    features_df = _add_lag_features(trajectories_in_splits)
    print(f"[bridge] feature table (post-lag, split-filtered): {len(features_df):,} rows")

    if args.dry_run:
        print("[bridge] dry-run complete; no files written")
        return 0

    if not args.no_backup:
        backup = _backup_legacy(args.processed)
        if backup:
            print(f"[bridge] HEE legacy backed up to {backup}")

    counts = _write_outputs(features_df, trajectories_df, scenarios_by_split, args.processed)
    print(f"[bridge] wrote outputs to {args.processed}")
    for name, n in counts.items():
        print(f"[bridge]   {name}: {n:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
