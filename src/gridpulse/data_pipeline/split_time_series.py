"""Time-series splitting utilities with calibration split and temporal gap support."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _estimate_gap_rows(df: pd.DataFrame, ts_col: str, gap_hours: int) -> int:
    if gap_hours <= 0 or ts_col not in df.columns:
        return max(0, int(gap_hours))
    ts = pd.to_datetime(df[ts_col], utc=True, errors="coerce").dropna()
    if len(ts) <= 1:
        return max(0, int(gap_hours))
    step_hours = ts.diff().dt.total_seconds().dropna().median() / 3600.0
    if not step_hours or step_hours <= 0:
        return max(0, int(gap_hours))
    return int(np.ceil(float(gap_hours) / float(step_hours)))


def time_split_with_calibration(
    df: pd.DataFrame,
    ts_col: str,
    train_ratio: float,
    calibration_ratio: float,
    val_ratio: float,
    gap_hours: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if train_ratio <= 0.0 or train_ratio >= 1.0:
        raise ValueError("train_ratio must be in (0, 1)")
    if calibration_ratio < 0.0 or calibration_ratio >= 1.0:
        raise ValueError("calibration_ratio must be in [0, 1)")
    if val_ratio < 0.0 or val_ratio >= 1.0:
        raise ValueError("val_ratio must be in [0, 1)")
    if gap_hours < 0:
        raise ValueError("gap_hours must be >= 0")

    total = train_ratio + calibration_ratio + val_ratio
    if total >= 1.0:
        raise ValueError("train_ratio + calibration_ratio + val_ratio must be < 1.0")

    n = len(df)
    n_train = int(n * train_ratio)
    n_cal = int(n * calibration_ratio)
    n_val = int(n * val_ratio)
    gap_rows = _estimate_gap_rows(df, ts_col=ts_col, gap_hours=gap_hours)

    train_start = 0
    train_end = n_train
    cal_start = min(n, train_end + gap_rows)
    cal_end = min(n, cal_start + n_cal)
    val_start = min(n, cal_end + gap_rows)
    val_end = min(n, val_start + n_val)
    test_start = min(n, val_end + gap_rows)

    train = df.iloc[train_start:train_end].copy()
    calibration = df.iloc[cal_start:cal_end].copy()
    val = df.iloc[val_start:val_end].copy()
    test = df.iloc[test_start:].copy()
    return train, calibration, val, test


def time_split(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    calibration_ratio: float = 0.0,
    gap_hours: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compatibility wrapper preserved for existing callers/tests."""
    ts_col = "timestamp" if "timestamp" in df.columns else df.columns[0]
    return time_split_with_calibration(
        df=df,
        ts_col=ts_col,
        train_ratio=train_ratio,
        calibration_ratio=calibration_ratio,
        val_ratio=val_ratio,
        gap_hours=gap_hours,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_path", required=True, help="Input features parquet/csv")
    p.add_argument("--out", dest="out_dir", default="data/processed/splits", help="Output directory")
    p.add_argument("--timestamp-col", default="timestamp")
    p.add_argument("--train-ratio", type=float, default=0.70)
    p.add_argument("--calibration-ratio", type=float, default=0.05)
    p.add_argument("--val-ratio", type=float, default=0.10)
    p.add_argument("--gap-hours", type=int, default=24)
    args = p.parse_args()

    in_path = Path(args.in_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if in_path.suffix == ".parquet":
        df = pd.read_parquet(in_path)
    else:
        df = pd.read_csv(in_path)

    if args.timestamp_col not in df.columns:
        raise ValueError(f"timestamp column '{args.timestamp_col}' not found in {in_path}")

    df[args.timestamp_col] = pd.to_datetime(df[args.timestamp_col], utc=True, errors="coerce")
    df = df.sort_values(args.timestamp_col).reset_index(drop=True)

    train, calibration, val, test = time_split_with_calibration(
        df=df,
        ts_col=args.timestamp_col,
        train_ratio=args.train_ratio,
        calibration_ratio=args.calibration_ratio,
        val_ratio=args.val_ratio,
        gap_hours=args.gap_hours,
    )

    train.to_parquet(out_dir / "train.parquet", index=False)
    calibration.to_parquet(out_dir / "calibration.parquet", index=False)
    val.to_parquet(out_dir / "val.parquet", index=False)
    test.to_parquet(out_dir / "test.parquet", index=False)

    (out_dir / "SPLIT_SUMMARY.md").write_text(
        f"""# Time Split Summary (with Calibration + Gap)

Input: `{in_path}`

Ratios:
- Train: {args.train_ratio}
- Calibration: {args.calibration_ratio}
- Validation: {args.val_ratio}
- Test: {1 - args.train_ratio - args.calibration_ratio - args.val_ratio}

Gap hours: {args.gap_hours}

| Split | Rows | Start | End |
|---|---:|---|---|
| Train | {len(train)} | {train[args.timestamp_col].min()} | {train[args.timestamp_col].max()} |
| Calibration | {len(calibration)} | {calibration[args.timestamp_col].min() if len(calibration) else "NA"} | {calibration[args.timestamp_col].max() if len(calibration) else "NA"} |
| Validation | {len(val)} | {val[args.timestamp_col].min() if len(val) else "NA"} | {val[args.timestamp_col].max() if len(val) else "NA"} |
| Test | {len(test)} | {test[args.timestamp_col].min() if len(test) else "NA"} | {test[args.timestamp_col].max() if len(test) else "NA"} |
""",
        encoding="utf-8",
    )

    print(f"Wrote splits to: {out_dir}")


if __name__ == "__main__":
    main()
