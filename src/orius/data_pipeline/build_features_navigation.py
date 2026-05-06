"""Build features for the navigation domain from ORIUS-format trajectory CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from orius.data_pipeline.split_time_series import time_split_with_calibration

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RAW = REPO_ROOT / "data" / "navigation" / "processed" / "navigation_orius.csv"
DEFAULT_OUT = REPO_ROOT / "data" / "navigation" / "processed"
TARGET = "vx"
LAG_STEPS = [1, 2, 4, 8, 12, 24]


def build_features(
    csv_path: Path,
    out_dir: Path,
    *,
    train_ratio: float = 0.70,
    calibration_ratio: float = 0.05,
    val_ratio: float = 0.10,
    gap_hours: float = 0.0,
) -> Path:
    """Build navigation features and temporal splits."""
    df = pd.read_csv(csv_path)
    for col in ("x", "y", "vx", "vy"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["x", "y", "vx", "vy"]).copy()
    df["timestamp"] = pd.to_datetime(df["ts_utc"], errors="coerce", utc=True)
    df = df.dropna(subset=["timestamp"]).sort_values(["robot_id", "step"]).reset_index(drop=True)

    # Use one robot trajectory for a clean single-series forecasting surface.
    robot_id = df["robot_id"].iloc[0]
    nav = df[df["robot_id"] == robot_id].copy().sort_values("step").reset_index(drop=True)

    for lag in LAG_STEPS:
        nav[f"x_lag{lag}"] = nav["x"].shift(lag)
        nav[f"y_lag{lag}"] = nav["y"].shift(lag)
        nav[f"vx_lag{lag}"] = nav["vx"].shift(lag)
        nav[f"vy_lag{lag}"] = nav["vy"].shift(lag)

    nav["hour"] = nav["timestamp"].dt.hour
    nav["minute"] = nav["timestamp"].dt.minute
    nav = nav.dropna().reset_index(drop=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    features_path = out_dir / "features.parquet"
    nav.to_parquet(features_path, index=False)

    splits_dir = out_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    train, cal, val, test = time_split_with_calibration(
        nav,
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
    parser = argparse.ArgumentParser(description="Build navigation features for training")
    parser.add_argument("--in", dest="input", type=Path, default=DEFAULT_RAW, help="Input CSV")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output directory")
    args = parser.parse_args()
    if not args.input.exists():
        return 1
    build_features(args.input, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
