#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from orius.forecasting.uncertainty.shift_aware import write_comparison_package


def _require_columns(path: Path, required: set[str]) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"missing input: {path}")
    df = pd.read_csv(path)
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"{path} missing columns: {sorted(missing)}")
    return df


def main() -> None:
    p = argparse.ArgumentParser(description="Build final acceptance package for shift-aware uncertainty")
    p.add_argument("--legacy-csv", required=True)
    p.add_argument("--shift-csv", required=True)
    p.add_argument("--out-dir", default="reports/publication/acceptance")
    p.add_argument("--target-coverage", type=float, default=0.9)
    p.add_argument("--max-width-increase", type=float, default=5.0)
    args = p.parse_args()

    required = {"y_true", "lower", "upper"}
    _require_columns(Path(args.legacy_csv), required)
    _require_columns(Path(args.shift_csv), required)

    signoff = write_comparison_package(
        legacy_csv=args.legacy_csv,
        shift_csv=args.shift_csv,
        out_dir=args.out_dir,
        target_coverage=args.target_coverage,
        max_width_increase=args.max_width_increase,
    )
    print(f"acceptance_signoff={Path(args.out_dir) / 'acceptance_signoff.json'}")
    print(f"all_checks_pass={signoff['all_checks_pass']}")


if __name__ == "__main__":
    main()
