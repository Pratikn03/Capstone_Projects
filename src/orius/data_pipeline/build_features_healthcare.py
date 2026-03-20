"""Build features for Healthcare domain from ORIUS vital signs CSV.

Reads data/healthcare/processed/healthcare_orius.csv, adds lag features,
produces features.parquet and splits for training.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from orius.data_pipeline.split_time_series import time_split_with_calibration


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RAW = REPO_ROOT / "data" / "healthcare" / "processed" / "healthcare_orius.csv"
DEFAULT_OUT = REPO_ROOT / "data" / "healthcare" / "processed"
TARGET = "hr_bpm"
LAG_STEPS = [1, 2, 4, 8, 12, 24]


def build_features(
    csv_path: Path,
    out_dir: Path,
    *,
    train_ratio: float = 0.70,
    calibration_ratio: float = 0.05,
    val_ratio: float = 0.10,
    gap_hours: float = 0.001,
) -> Path:
    """Build Healthcare features and splits."""
    df = pd.read_csv(csv_path)
    for c in ["hr_bpm", "spo2_pct", "respiratory_rate"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["hr_bpm"]).reset_index(drop=True)

    if "ts_utc" in df.columns:
        df["timestamp"] = pd.to_datetime(df["ts_utc"], errors="coerce")
    elif "Time [s]" in df.columns:
        df["timestamp"] = pd.to_datetime("2026-01-01") + pd.to_timedelta(df["Time [s]"], unit="s")
    else:
        df["timestamp"] = pd.to_datetime("2026-01-01") + pd.to_timedelta(df["step"], unit="s")
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Convert multi-patient rows into one timestamp-aligned monitoring stream.
    agg_spec: dict[str, str] = {"hr_bpm": "mean"}
    if "spo2_pct" in df.columns:
        agg_spec["spo2_pct"] = "mean"
    if "respiratory_rate" in df.columns:
        agg_spec["respiratory_rate"] = "mean"
    df = df.groupby("timestamp", as_index=False).agg(agg_spec)
    df["ts_utc"] = df["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    for lag in LAG_STEPS:
        df[f"hr_bpm_lag{lag}"] = df["hr_bpm"].shift(lag)
        if "spo2_pct" in df.columns:
            df[f"spo2_pct_lag{lag}"] = df["spo2_pct"].shift(lag)
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
        gap_hours=0,
    )
    train.to_parquet(splits_dir / "train.parquet", index=False)
    cal.to_parquet(splits_dir / "calibration.parquet", index=False)
    val.to_parquet(splits_dir / "val.parquet", index=False)
    test.to_parquet(splits_dir / "test.parquet", index=False)

    return features_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Healthcare features for training")
    parser.add_argument("--in", dest="input", type=Path, default=DEFAULT_RAW, help="Input CSV")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output directory")
    args = parser.parse_args()
    if not args.input.exists():
        print(f"Input not found: {args.input}. Run: make healthcare-datasets")
        return 1
    build_features(args.input, args.out)
    print(f"Healthcare features -> {args.out / 'features.parquet'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
