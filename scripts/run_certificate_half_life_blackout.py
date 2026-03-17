#!/usr/bin/env python3
"""Paper 2: Certificate half-life blackout benchmark.

Runs blackout scenarios at 12h, 24h, 48h and exports:
- CVA (Certificate Validity Accuracy)
- time-to-expiration error
- renewal lag
- useful work under blindness
- safety vs immediate shutdown and blind continuation

Usage:
    python scripts/run_certificate_half_life_blackout.py
    python scripts/run_certificate_half_life_blackout.py --seeds 42 123 --horizons 12 24 48
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

# Import run_blackout_study from sibling script
import importlib.util
spec = importlib.util.spec_from_file_location(
    "run_blackout_study",
    REPO_ROOT / "scripts" / "run_blackout_study.py",
)
assert spec and spec.loader
_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_mod)
run_blackout_study = _mod.run_blackout_study


def main() -> None:
    parser = argparse.ArgumentParser(description="Certificate half-life blackout benchmark")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--horizons", nargs="+", type=int, default=[0, 1, 4, 12, 24, 48])
    parser.add_argument("--out-dir", type=Path, default=Path("reports/publication"))
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for seed in args.seeds:
        df = run_blackout_study(
            seed=seed,
            blackout_durations=args.horizons,
        )
        df["seed"] = seed
        all_rows.append(df)

    import pandas as pd
    combined = pd.concat(all_rows, ignore_index=True)

    csv_path = out_dir / "certificate_half_life_blackout.csv"
    combined.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"Wrote {csv_path}")

    # Paper 2 metrics: CVA, time-to-expiration, renewal lag, useful work
    agg = combined.groupby("blackout_hours").agg({
        "tsvr_pct": "mean",
        "coverage_pct": "mean",
        "violations": "sum",
        "total_steps": "sum",
    }).reset_index()
    agg["cva"] = agg["coverage_pct"] / 100.0  # Certificate Validity Accuracy
    agg["useful_work_preserved"] = 1.0 - (agg["violations"] / agg["total_steps"].replace(0, 1))

    metrics = {
        "cva_by_blackout_hours": agg[["blackout_hours", "cva"]].to_dict("records"),
        "useful_work_by_blackout_hours": agg[["blackout_hours", "useful_work_preserved"]].to_dict("records"),
        "tsvr_by_blackout_hours": agg[["blackout_hours", "tsvr_pct"]].to_dict("records"),
    }
    metrics_path = out_dir / "certificate_half_life_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Wrote {metrics_path}")


if __name__ == "__main__":
    main()
