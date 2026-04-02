"""Build features for AV (autonomous vehicle) domain from ORIUS trajectory CSV.

Reads data/av/processed/av_trajectories_orius.csv, adds lag features,
produces features.parquet and splits for training.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from orius.data_pipeline.split_time_series import time_split_with_calibration


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RAW = REPO_ROOT / "data" / "av" / "processed" / "av_trajectories_orius.csv"
DEFAULT_OUT = REPO_ROOT / "data" / "av" / "processed"
TARGET = "speed_mps"
LAG_STEPS = [1, 2, 4, 8, 12, 24]


def build_features(
    csv_path: Path,
    out_dir: Path,
    *,
    train_ratio: float = 0.70,
    calibration_ratio: float = 0.05,
    val_ratio: float = 0.10,
    gap_hours: float = 0.01,
) -> Path:
    """Build AV features and splits."""
    df = pd.read_csv(csv_path)
    for col in ["position_m", "speed_mps", "speed_limit_mps"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["speed_mps", "position_m"])
    df = df.sort_values(["vehicle_id", "step"]).reset_index(drop=True)

    # Use first vehicle for a clean time series (or aggregate)
    v0 = df[df["vehicle_id"] == df["vehicle_id"].iloc[0]].copy()
    v0 = v0.sort_values("step").reset_index(drop=True)

    # Prefer explicit timestamps from real corpora; keep a synthetic fallback.
    if "ts_utc" in v0.columns:
        v0["timestamp"] = pd.to_datetime(v0["ts_utc"], errors="coerce", utc=True)
    else:
        v0["timestamp"] = pd.to_datetime("2026-01-01") + pd.to_timedelta(v0["step"] * 0.1, unit="s")
    v0 = v0.dropna(subset=["timestamp"]).reset_index(drop=True)

    # Lag features
    for lag in LAG_STEPS:
        v0[f"speed_mps_lag{lag}"] = v0["speed_mps"].shift(lag)
        v0[f"position_m_lag{lag}"] = v0["position_m"].astype(float).shift(lag)
    v0["speed_limit_mps"] = v0["speed_limit_mps"].fillna(30.0)
    v0["hour"] = v0["timestamp"].dt.hour
    v0["minute"] = v0["timestamp"].dt.minute

    v0 = v0.dropna().reset_index(drop=True)
    features_path = out_dir / "features.parquet"
    out_dir.mkdir(parents=True, exist_ok=True)
    v0.to_parquet(features_path, index=False)

    # Splits
    splits_dir = out_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    train, cal, val, test = time_split_with_calibration(
        v0,
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Build AV features for training")
    parser.add_argument("--in", dest="input", type=Path, default=DEFAULT_RAW, help="Input CSV")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output directory")
    args = parser.parse_args()
    if not args.input.exists():
        print(f"Input not found: {args.input}. Run: make av-datasets")
        return 1
    build_features(args.input, args.out)
    print(f"AV features -> {args.out / 'features.parquet'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
