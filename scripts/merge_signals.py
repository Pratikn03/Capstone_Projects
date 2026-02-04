"""Merge carbon intensity and MOER signals into a single file."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _load(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--carbon", required=True, help="Carbon signals CSV/Parquet (timestamp, carbon_kg_per_mwh)")
    ap.add_argument("--moer", required=True, help="MOER signals CSV/Parquet (timestamp, moer_kg_per_mwh)")
    ap.add_argument("--out", default="data/raw/price_carbon_signals.csv")
    args = ap.parse_args()

    carbon = _load(Path(args.carbon))
    moer = _load(Path(args.moer))
    merged = pd.merge(carbon, moer, on="timestamp", how="outer").sort_values("timestamp")
    merged = merged.dropna(subset=["carbon_kg_per_mwh", "moer_kg_per_mwh"], how="all")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(merged)} rows)")


if __name__ == "__main__":
    main()
