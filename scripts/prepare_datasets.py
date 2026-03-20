#!/usr/bin/env python3
"""Dataset preparation utility for ORIUS real-data validation.

Checks whether real datasets are present at the canonical paths. If absent,
generates calibrated synthetic fallback files from published distribution
statistics and writes them to the canonical paths so that the real-data
validation run can proceed.

Usage
-----
    python scripts/prepare_datasets.py [--force-synthetic]

Canonical paths
---------------
  data/ccpp/CCPP.csv     — UCI Combined Cycle Power Plant
  data/bidmc/bidmc_vitals.csv  — PhysioNet BIDMC ICU vitals

To use REAL data
----------------
1. UCI CCPP:
   - Download from https://archive.ics.uci.edu/ml/datasets/Combined+Cycle+Power+Plant
   - Extract Folds5x2_pp.xlsx, export first sheet as CSV with header AT,V,AP,RH,PE
   - Place at data/ccpp/CCPP.csv

2. PhysioNet BIDMC:
   - Download bidmc_csv.tar.gz from https://physionet.org/content/bidmc/1.0.0/
   - Extract bidmc_*_Numerics.csv files
   - Concatenate and keep columns HR, SpO2, RR; save as data/bidmc/bidmc_vitals.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from orius.orius_bench.real_data_loader import (
    BIDMC_PATH,
    CCPP_PATH,
    dataset_status,
    generate_bidmc_synthetic,
    generate_ccpp_synthetic,
)


def _write_ccpp(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["AT", "V", "AP", "RH", "PE"])
        writer.writeheader()
        writer.writerows(rows)


def _write_bidmc(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["HR", "SpO2", "RR"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare ORIUS real-data datasets")
    parser.add_argument(
        "--force-synthetic",
        action="store_true",
        help="Overwrite with synthetic data even if real files exist",
    )
    args = parser.parse_args()

    status = dataset_status()

    # ---- CCPP ----
    ccpp = status["ccpp"]
    if ccpp["real_data"] and not args.force_synthetic:
        print(f"[CCPP]  Real data found: {ccpp['path']} ({ccpp['rows']} rows)")
    else:
        reason = "force-synthetic" if args.force_synthetic else "not found"
        print(f"[CCPP]  Real data {reason}. Generating calibrated synthetic ({ccpp['fallback_rows']} rows)...")
        rows = generate_ccpp_synthetic(n=ccpp["fallback_rows"], seed=42)
        _write_ccpp(CCPP_PATH, rows)
        print(f"[CCPP]  Written to {CCPP_PATH}")

    # ---- BIDMC ----
    bidmc = status["bidmc"]
    if bidmc["real_data"] and not args.force_synthetic:
        print(f"[BIDMC] Real data found: {bidmc['path']} ({bidmc['rows']} rows)")
    else:
        reason = "force-synthetic" if args.force_synthetic else "not found"
        print(f"[BIDMC] Real data {reason}. Generating calibrated synthetic ({bidmc['fallback_rows']} rows)...")
        rows = generate_bidmc_synthetic(n=bidmc["fallback_rows"], seed=42)
        _write_bidmc(BIDMC_PATH, rows)
        print(f"[BIDMC] Written to {BIDMC_PATH}")

    # Verify
    status2 = dataset_status()
    ok = True
    for name, info in status2.items():
        label = "real" if info["real_data"] else "synthetic-fallback"
        print(f"\n  {name.upper():6s}: {info['rows']} rows ({label})")
        if info["rows"] == 0:
            print(f"  ERROR: {name} has 0 rows — validation will fail")
            ok = False

    print("\nDataset preparation complete." if ok else "\nERROR: some datasets are empty.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
