#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from orius.forecasting.uncertainty.shift_aware import SubgroupCoverageTracker, write_shift_aware_artifacts


def main() -> None:
    p = argparse.ArgumentParser(description="Run shift-aware conditional coverage audit")
    p.add_argument("--input", required=True, help="CSV with y_true,y_pred,lower,upper,reliability_w")
    p.add_argument("--out-dir", default="reports/publication")
    args = p.parse_args()

    path = Path(args.input)
    if not path.exists():
        raise SystemExit(f"missing input: {path}")
    df = pd.read_csv(path)
    required = {"y_true", "lower", "upper", "reliability_w"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"missing columns: {sorted(missing)}")

    tracker = SubgroupCoverageTracker(target_coverage=0.9)
    for _, r in df.iterrows():
        covered = bool(float(r["lower"]) <= float(r["y_true"]) <= float(r["upper"]))
        key = tracker.build_group_key(
            reliability_score=float(r["reliability_w"]),
            volatility=float(abs(r.get("volatility", 0.0))),
            fault_type=str(r.get("fault_type", "none")),
            ts=str(r.get("timestamp", "")),
            custom_key=str(r.get("custom_group_key", "")) or None,
        )
        tracker.update(
            group_key=key,
            covered=covered,
            interval_width=float(r["upper"] - r["lower"]),
            abs_residual=float(abs(r["y_true"] - r.get("y_pred", r["y_true"]))),
        )

    write_shift_aware_artifacts(
        tracker=tracker,
        validity_trace=[],
        adaptive_trace=[],
        publication_dir=args.out_dir,
    )
    print(f"rows={len(df)} groups={len(tracker.group_rows())} out={args.out_dir}")


if __name__ == "__main__":
    main()
