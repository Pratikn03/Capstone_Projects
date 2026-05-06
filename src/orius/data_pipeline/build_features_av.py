"""Build AV artifacts.

Two modes are supported:

1. Scenario-native Waymo validation mode.
   Input is the repo-local TFRecord shard directory under
   ``data/orius_av/raw/waymo_motion/validation`` and the output is the Stage 2
   validation surface in ``data/orius_av/av/processed``.

2. Legacy CSV mode.
   Input is a processed 1D trajectory CSV and the output is the older
   ``features.parquet`` plus time-based splits.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from orius.av_waymo import build_validation_surface
from orius.data_pipeline.split_time_series import time_split_with_calibration

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RAW = REPO_ROOT / "data" / "orius_av" / "raw" / "waymo_motion" / "validation"
DEFAULT_OUT = REPO_ROOT / "data" / "orius_av" / "av" / "processed"
TARGET = "speed_mps"
LAG_STEPS = [1, 2, 4, 8, 12, 24]


def _build_legacy_features(
    csv_path: Path,
    out_dir: Path,
    *,
    train_ratio: float = 0.70,
    calibration_ratio: float = 0.05,
    val_ratio: float = 0.10,
    gap_hours: float = 0.01,
) -> Path:
    """Build legacy 1D AV features and splits from a processed CSV."""
    df = pd.read_csv(csv_path)
    for col in [
        "position_m",
        "speed_mps",
        "speed_limit_mps",
        "lead_position_m",
        "lead_rel_x_m",
        "lead_speed_mps",
        "rss_safe_gap_m",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["speed_mps", "position_m"])
    df = df.sort_values(["vehicle_id", "step"]).reset_index(drop=True)

    # Prefer explicit timestamps from real corpora; keep a synthetic fallback.
    if "ts_utc" in df.columns:
        df["timestamp"] = pd.to_datetime(df["ts_utc"], errors="coerce", utc=True)
    else:
        df["timestamp"] = pd.to_datetime("2026-01-01") + pd.to_timedelta(df["step"] * 0.1, unit="s")
    df = df.dropna(subset=["timestamp"]).reset_index(drop=True)

    # Lag features are computed per vehicle so the bridged Waymo corpus
    # survives rebuilds instead of collapsing to a single ego trace.
    for lag in LAG_STEPS:
        df[f"speed_mps_lag{lag}"] = df.groupby("vehicle_id", sort=False)["speed_mps"].shift(lag)
        df[f"position_m_lag{lag}"] = df.groupby("vehicle_id", sort=False)["position_m"].shift(lag)
    df["speed_limit_mps"] = df["speed_limit_mps"].fillna(30.0)
    df["hour"] = df["timestamp"].dt.hour
    df["minute"] = df["timestamp"].dt.minute

    required_cols = [f"speed_mps_lag{lag}" for lag in LAG_STEPS] + [
        f"position_m_lag{lag}" for lag in LAG_STEPS
    ]
    for target_col in ("speed_mps", "position_m", "lead_position_m", "rss_safe_gap_m"):
        if target_col in df.columns:
            required_cols.append(target_col)
    df = df.dropna(subset=required_cols).reset_index(drop=True)
    features_path = out_dir / "features.parquet"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(features_path, index=False)

    # Splits
    splits_dir = out_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    train, cal, val, test = time_split_with_calibration(
        df,
        ts_col="timestamp",
        train_ratio=train_ratio,
        calibration_ratio=calibration_ratio,
        val_ratio=val_ratio,
        gap_hours=0,
    )
    train.to_parquet(splits_dir / "train.parquet", index=False)
    cal.to_parquet(splits_dir / "calibration.parquet", index=False)
    val.to_parquet(splits_dir / "val.parquet", index=False)
    test.to_parquet(splits_dir / "test.parquet", index=False)

    return features_path


def build_features(
    input_path: Path,
    out_dir: Path,
    *,
    train_ratio: float = 0.70,
    calibration_ratio: float = 0.05,
    val_ratio: float = 0.10,
    gap_hours: float = 0.01,
    max_shards: int | None = None,
    max_scenarios: int | None = None,
    skip_actor_tracks: bool = False,
) -> Path:
    """Dispatch to the scenario-native or legacy AV builder."""
    if input_path.is_dir() or input_path.suffix.startswith(".tfrecord"):
        report = build_validation_surface(
            raw_dir=input_path,
            out_dir=out_dir,
            max_shards=max_shards,
            max_scenarios=max_scenarios,
            write_actor_tracks=not skip_actor_tracks,
        )
        return Path(report["artifacts"]["scenario_index"])
    return _build_legacy_features(
        input_path,
        out_dir,
        train_ratio=train_ratio,
        calibration_ratio=calibration_ratio,
        val_ratio=val_ratio,
        gap_hours=gap_hours,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Build AV features or Waymo validation artifacts")
    parser.add_argument(
        "--in", dest="input", type=Path, default=DEFAULT_RAW, help="Input CSV or raw TFRecord directory"
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output directory")
    parser.add_argument(
        "--max-shards", type=int, default=None, help="Limit raw shard scans in scenario-native mode"
    )
    parser.add_argument(
        "--max-scenarios", type=int, default=None, help="Limit scenario scans in scenario-native mode"
    )
    parser.add_argument(
        "--skip-actor-tracks", action="store_true", help="Skip actor_tracks.parquet in scenario-native mode"
    )
    args = parser.parse_args()
    if not args.input.exists():
        return 1
    build_features(
        args.input,
        args.out,
        max_shards=args.max_shards,
        max_scenarios=args.max_scenarios,
        skip_actor_tracks=args.skip_actor_tracks,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
