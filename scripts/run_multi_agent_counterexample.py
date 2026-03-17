#!/usr/bin/env python3
"""Multi-agent non-composition counterexample runner (Paper 5).

Demonstrates that local DC3S certificates do NOT compose when agents
share feeder capacity. Outputs CSV and summary JSON.

Usage:
    python scripts/run_multi_agent_counterexample.py [--out reports/multi_agent]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from orius.multi_agent.scenarios import run_transformer_capacity_scenario


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="reports/multi_agent", help="Output directory")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    results = run_transformer_capacity_scenario(out_dir=str(out))

    # Summary
    summary = {}
    for proto, data in results.items():
        summary[proto] = {
            "joint_violations": data.get("joint_violations", 0),
            "useful_work": data.get("useful_work", 0),
            "fairness": data.get("fairness", 0),
        }

    summary_path = out / "counterexample_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("=== Multi-Agent Non-Composition Counterexample ===")
    for proto, s in summary.items():
        print(f"  {proto:30s} | violations={s['joint_violations']} "
              f"| useful_work={s['useful_work']:.1f} | fairness={s['fairness']:.4f}")
    print(f"\nSummary → {summary_path}")


if __name__ == "__main__":
    main()
