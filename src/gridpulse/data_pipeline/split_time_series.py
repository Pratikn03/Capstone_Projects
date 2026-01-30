from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

def time_split(df: pd.DataFrame, train_ratio: float = 0.7, val_ratio: float = 0.15):
    """Time-ordered split: train earliest, then val, then test."""
    n = len(df)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = df.iloc[:n_train].copy()
    val = df.iloc[n_train:n_train+n_val].copy()
    test = df.iloc[n_train+n_val:].copy()
    return train, val, test

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_path", required=True, help="Input parquet (data/processed/features.parquet)")
    p.add_argument("--out", dest="out_dir", default="data/processed/splits", help="Output directory")
    p.add_argument("--timestamp-col", default="timestamp")
    p.add_argument("--train-ratio", type=float, default=0.7)
    p.add_argument("--val-ratio", type=float, default=0.15)
    args = p.parse_args()

    in_path = Path(args.in_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(in_path)
    if args.timestamp_col not in df.columns:
        raise ValueError(f"timestamp column '{args.timestamp_col}' not found in {in_path}")
    df[args.timestamp_col] = pd.to_datetime(df[args.timestamp_col], utc=True, errors="coerce")
    df = df.sort_values(args.timestamp_col).reset_index(drop=True)

    train, val, test = time_split(df, args.train_ratio, args.val_ratio)

    train.to_parquet(out_dir / "train.parquet", index=False)
    val.to_parquet(out_dir / "val.parquet", index=False)
    test.to_parquet(out_dir / "test.parquet", index=False)

    (out_dir / "SPLIT_SUMMARY.md").write_text(
        f"""# Time Split Summary

Input: `{in_path}`

- Train ratio: {args.train_ratio}
- Val ratio: {args.val_ratio}
- Test ratio: {1 - args.train_ratio - args.val_ratio}

| Split | Rows | Start | End |
|---|---:|---|---|
| Train | {len(train)} | {train[args.timestamp_col].min()} | {train[args.timestamp_col].max()} |
| Val | {len(val)} | {val[args.timestamp_col].min()} | {val[args.timestamp_col].max()} |
| Test | {len(test)} | {test[args.timestamp_col].min()} | {test[args.timestamp_col].max()} |
""",
        encoding="utf-8"
    )

    print(f"Wrote splits to: {out_dir}")

if __name__ == "__main__":
    main()
