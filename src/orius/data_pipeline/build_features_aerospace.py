"""Build features for Aerospace domain from ORIUS flight CSV.

Reads data/aerospace/processed/aerospace_orius.csv, adds lag features,
produces features.parquet and splits for training.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from orius.data_pipeline.split_time_series import time_split_with_calibration


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RAW = REPO_ROOT / "data" / "aerospace" / "processed" / "aerospace_orius.csv"
DEFAULT_OUT = REPO_ROOT / "data" / "aerospace" / "processed"
TARGET = "airspeed_kt"
LAG_STEPS = [1, 2, 4, 8, 12, 24]


def build_features(
    csv_path: Path,
    out_dir: Path,
    *,
    train_ratio: float = 0.70,
    calibration_ratio: float = 0.05,
    val_ratio: float = 0.10,
    gap_hours: int = 0,
) -> Path:
    """Build Aerospace features and splits."""
    df = pd.read_csv(csv_path)
    for col in ["altitude_m", "airspeed_kt", "bank_angle_deg", "fuel_remaining_pct"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["airspeed_kt"]).reset_index(drop=True)

    df["timestamp"] = pd.to_datetime(df["ts_utc"], errors="coerce")
    df = df.sort_values("timestamp").reset_index(drop=True)

    for lag in LAG_STEPS:
        df[f"airspeed_kt_lag{lag}"] = df["airspeed_kt"].shift(lag)
        df[f"altitude_m_lag{lag}"] = df["altitude_m"].shift(lag)
    df["hour"] = df["timestamp"].dt.hour
    df["minute"] = df["timestamp"].dt.minute

    df = df.dropna().reset_index(drop=True)
    features_path = out_dir / "features.parquet"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(features_path, index=False)

    splits_dir = out_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    train, cal, val, test = time_split_with_calibration(
        df,
        ts_col="timestamp",
        train_ratio=train_ratio,
        calibration_ratio=calibration_ratio,
        val_ratio=val_ratio,
        gap_hours=gap_hours,
    )
    train.to_parquet(splits_dir / "train.parquet", index=False)
    cal.to_parquet(splits_dir / "calibration.parquet", index=False)
    val.to_parquet(splits_dir / "val.parquet", index=False)
    test.to_parquet(splits_dir / "test.parquet", index=False)

    return features_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Aerospace features for training")
    parser.add_argument("--in", dest="input", type=Path, default=DEFAULT_RAW, help="Input CSV")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output directory")
    args = parser.parse_args()
    if not args.input.exists():
        print(f"Input not found: {args.input}. Run: python scripts/download_aerospace_datasets.py")
        return 1
    build_features(args.input, args.out)
    print(f"Aerospace features -> {args.out / 'features.parquet'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
