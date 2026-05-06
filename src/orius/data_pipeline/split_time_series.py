from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _next_start_idx(
    df: pd.DataFrame,
    ts_col: str,
    start_idx: int,
    anchor_ts: pd.Timestamp,
    gap: pd.Timedelta,
) -> int:
    idx = int(start_idx)
    threshold = anchor_ts + gap
    # strict holdout buffer: enforce > (anchor + gap)
    while idx < len(df) and df.iloc[idx][ts_col] <= threshold:
        idx += 1
    return idx


def time_split_with_calibration(
    df: pd.DataFrame,
    ts_col: str,
    train_ratio: float,
    calibration_ratio: float,
    val_ratio: float,
    gap_hours: int,
):
    if train_ratio <= 0 or val_ratio <= 0 or calibration_ratio <= 0:
        raise ValueError("train_ratio, calibration_ratio, val_ratio must be > 0")
    if train_ratio + calibration_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + calibration_ratio + val_ratio must be < 1.0")

    n = len(df)
    n_train = int(n * train_ratio)
    n_cal = int(n * calibration_ratio)
    n_val = int(n * val_ratio)

    train = df.iloc[:n_train].copy()
    cursor = n_train
    gap = pd.Timedelta(hours=int(gap_hours))

    if gap_hours > 0 and not train.empty:
        cursor = _next_start_idx(df, ts_col, cursor, train[ts_col].max(), gap)

    cal = df.iloc[cursor : cursor + n_cal].copy()
    cursor += len(cal)

    if gap_hours > 0 and not cal.empty:
        cursor = _next_start_idx(df, ts_col, cursor, cal[ts_col].max(), gap)

    val = df.iloc[cursor : cursor + n_val].copy()
    cursor += len(val)

    if gap_hours > 0 and not val.empty:
        cursor = _next_start_idx(df, ts_col, cursor, val[ts_col].max(), gap)

    test = df.iloc[cursor:].copy()

    return train, cal, val, test


def time_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    calibration_ratio: float = 0.0,
    gap_hours: int = 0,
    ts_col: str = "timestamp",
):
    """Compatibility wrapper used by legacy callers/tests.

    Signature is preserved as:
    `time_split(df, train_ratio, val_ratio, calibration_ratio, gap_hours)`.
    """
    frame = df.copy()
    if ts_col not in frame.columns:
        raise ValueError(f"timestamp column '{ts_col}' not found")

    frame[ts_col] = pd.to_datetime(frame[ts_col], utc=True, errors="coerce")
    frame = frame.sort_values(ts_col).reset_index(drop=True)

    if calibration_ratio < 0 or val_ratio <= 0 or train_ratio <= 0:
        raise ValueError("train_ratio and val_ratio must be > 0, calibration_ratio must be >= 0")

    if calibration_ratio == 0.0:
        if train_ratio + val_ratio >= 1.0:
            raise ValueError("train_ratio + val_ratio must be < 1.0")

        n = len(frame)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train = frame.iloc[:n_train].copy()
        calibration = frame.iloc[0:0].copy()
        cursor = n_train
        if gap_hours > 0 and not train.empty:
            gap = pd.Timedelta(hours=int(gap_hours))
            cursor = _next_start_idx(frame, ts_col, cursor, train[ts_col].max(), gap)
        val = frame.iloc[cursor : cursor + n_val].copy()
        cursor += len(val)
        if gap_hours > 0 and not val.empty:
            gap = pd.Timedelta(hours=int(gap_hours))
            cursor = _next_start_idx(frame, ts_col, cursor, val[ts_col].max(), gap)
        test = frame.iloc[cursor:].copy()

        return train, calibration, val, test

    return time_split_with_calibration(
        df=frame,
        ts_col=ts_col,
        train_ratio=train_ratio,
        calibration_ratio=calibration_ratio,
        val_ratio=val_ratio,
        gap_hours=gap_hours,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_path", required=True)
    p.add_argument("--out", dest="out_dir", default="data/processed/splits")
    p.add_argument("--timestamp-col", default="timestamp")
    p.add_argument("--train-ratio", type=float, default=0.70)
    p.add_argument("--calibration-ratio", type=float, default=0.05)
    p.add_argument("--val-ratio", type=float, default=0.10)
    p.add_argument("--gap-hours", type=int, default=24)
    args = p.parse_args()

    in_path = Path(args.in_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(in_path) if in_path.suffix == ".parquet" else pd.read_csv(in_path)
    if args.timestamp_col not in df.columns:
        raise ValueError(f"timestamp column '{args.timestamp_col}' not found")

    df[args.timestamp_col] = pd.to_datetime(df[args.timestamp_col], utc=True, errors="coerce")
    df = df.sort_values(args.timestamp_col).reset_index(drop=True)

    train, cal, val, test = time_split_with_calibration(
        df=df,
        ts_col=args.timestamp_col,
        train_ratio=args.train_ratio,
        calibration_ratio=args.calibration_ratio,
        val_ratio=args.val_ratio,
        gap_hours=args.gap_hours,
    )

    train.to_parquet(out_dir / "train.parquet", index=False)
    cal.to_parquet(out_dir / "calibration.parquet", index=False)
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
| Calibration | {len(cal)} | {cal[args.timestamp_col].min()} | {cal[args.timestamp_col].max()} |
| Validation | {len(val)} | {val[args.timestamp_col].min()} | {val[args.timestamp_col].max()} |
| Test | {len(test)} | {test[args.timestamp_col].min()} | {test[args.timestamp_col].max()} |
""",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
