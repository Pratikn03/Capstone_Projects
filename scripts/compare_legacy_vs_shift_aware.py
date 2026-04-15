#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from orius.forecasting.uncertainty.shift_aware import write_comparison_package


def main() -> None:
    p = argparse.ArgumentParser(description="Compare legacy vs shift-aware interval evidence")
    p.add_argument("--legacy-csv", required=True)
    p.add_argument("--shift-csv", required=True)
    p.add_argument("--out-dir", default="reports/publication")
    p.add_argument("--target-coverage", type=float, default=0.9)
    p.add_argument("--max-width-increase", type=float, default=5.0)
    args = p.parse_args()

    for path in [args.legacy_csv, args.shift_csv]:
        if not Path(path).exists():
            raise SystemExit(f"missing input: {path}")

    signoff = write_comparison_package(
        legacy_csv=args.legacy_csv,
        shift_csv=args.shift_csv,
        out_dir=args.out_dir,
        target_coverage=args.target_coverage,
        max_width_increase=args.max_width_increase,
    )
    print(f"all_checks_pass={signoff['all_checks_pass']} out_dir={args.out_dir}")


if __name__ == "__main__":
    main()
