#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from orius.forecasting.uncertainty.shift_aware import SubgroupCoverageTracker, ShiftAwareConfig, write_shift_aware_artifacts


def main() -> None:
    ap = argparse.ArgumentParser(description="Run shift-aware subgroup coverage audit.")
    ap.add_argument("--input", required=True, help="CSV with y_true,y_hat,half_width,reliability,volatility,fault_type,hour")
    ap.add_argument("--out-dir", default="reports/publication")
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise SystemExit(f"missing input: {in_path}")

    tracker = SubgroupCoverageTracker(config=ShiftAwareConfig(enabled=True), target_coverage=0.9)
    total = 0
    covered_total = 0
    with in_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            y_true = float(row["y_true"])
            y_hat = float(row["y_hat"])
            half = float(row["half_width"])
            covered = (y_hat - half) <= y_true <= (y_hat + half)
            covered_total += int(covered)
            total += 1
            tracker.update(
                covered=covered,
                interval_width=2.0 * half,
                abs_residual=abs(y_true - y_hat),
                context={
                    "reliability": float(row.get("reliability", 1.0)),
                    "volatility": float(row.get("volatility", 0.0)),
                    "fault_type": row.get("fault_type", "none"),
                    "hour": int(float(row.get("hour", 0))),
                    "custom_group": row.get("custom_group", "default"),
                },
            )

    rows = tracker.to_rows()
    out = write_shift_aware_artifacts(
        reliability_rows=rows,
        volatility_rows=rows,
        fault_rows=rows,
        validity_trace_rows=[],
        quantile_trace_rows=[],
        out_dir=args.out_dir,
    )
    cov = covered_total / max(1, total)
    print(f"marginal_coverage={cov:.4f} n={total}")
    print(f"wrote={out}")


if __name__ == "__main__":
    main()
