"""Build features for Industrial domain from ORIUS process CSV.

Reads data/industrial/processed/industrial_orius.csv, adds lag features,
produces features.parquet and splits for training.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from orius.data_pipeline.split_time_series import time_split_with_calibration


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RAW = REPO_ROOT / "data" / "industrial" / "processed" / "industrial_orius.csv"
DEFAULT_OUT = REPO_ROOT / "data" / "industrial" / "processed"
TARGET = "power_mw"
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
    """Build Industrial features and splits."""
    df = pd.read_csv(csv_path)
    for col in ["temp_c", "pressure_mbar", "humidity_pct", "power_mw"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["power_mw"]).reset_index(drop=True)

    if "ts_utc" in df.columns:
        df["timestamp"] = pd.to_datetime(df["ts_utc"], errors="coerce")
    else:
        df["timestamp"] = pd.to_datetime("2026-01-01") + pd.to_timedelta(df["step"], unit="h")
    df = df.sort_values("timestamp").reset_index(drop=True)

    for lag in LAG_STEPS:
        df[f"power_mw_lag{lag}"] = df["power_mw"].shift(lag)
        if "temp_c" in df.columns:
            df[f"temp_c_lag{lag}"] = df["temp_c"].shift(lag)
    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek

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
    parser = argparse.ArgumentParser(description="Build Industrial features for training")
    parser.add_argument("--in", dest="input", type=Path, default=DEFAULT_RAW, help="Input CSV")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output directory")
    args = parser.parse_args()
    if not args.input.exists():
        print(f"Input not found: {args.input}. Run: make industrial-datasets")
        return 1
    build_features(args.input, args.out)
    print(f"Industrial features -> {args.out / 'features.parquet'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
