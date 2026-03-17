#!/usr/bin/env python3
"""Build Paper 2 half-life tables from blackout benchmark outputs.

Usage:
    python scripts/build_half_life_tables.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd


def main() -> None:
    in_path = REPO_ROOT / "reports/publication/certificate_half_life_blackout.csv"
    if not in_path.exists():
        in_path = REPO_ROOT / "reports/publication/blackout_study.csv"
    if not in_path.exists():
        print("Run scripts/run_certificate_half_life_blackout.py first.")
        sys.exit(1)

    out_dir = REPO_ROOT / "reports/publication"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)
    if "blackout_hours" not in df.columns and "blackout_hours" in str(in_path):
        if "blackout_hours" in df.columns:
            pass
        else:
            print("Unexpected CSV format")
            sys.exit(1)

    agg = df.groupby("blackout_hours", as_index=False).agg({
        "tsvr_pct": "mean",
        "coverage_pct": "mean",
        "violations": "sum",
        "total_steps": "sum",
    })
    agg["cva"] = (agg["coverage_pct"] / 100.0).round(4)
    agg["tsvr"] = (agg["tsvr_pct"] / 100.0).round(4)

    table_path = out_dir / "tbl_half_life_blackout.csv"
    agg[["blackout_hours", "cva", "tsvr", "violations", "total_steps"]].to_csv(
        table_path, index=False, float_format="%.4f"
    )
    print(f"Wrote {table_path}")


if __name__ == "__main__":
    main()
