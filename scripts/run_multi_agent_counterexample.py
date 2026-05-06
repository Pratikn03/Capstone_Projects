#!/usr/bin/env python3
"""Multi-agent non-composition counterexample runner for bounded composition.

Demonstrates that local DC3S certificates do NOT compose when agents
share feeder capacity. Outputs CSV and summary JSON.

Usage:
    python scripts/run_multi_agent_counterexample.py [--out reports/multi_agent]
"""

from __future__ import annotations

import argparse
import csv
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
            "useful_work": data.get("useful_work_mwh", data.get("useful_work", 0)),
            "fairness": data.get("fairness", 0),
        }

    summary_path = out / "counterexample_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # protocol_compare.csv (clean comparison artifact)
    proto_compare_path = out / "protocol_compare.csv"
    with open(proto_compare_path, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "protocol",
                "joint_violations",
                "local_violations",
                "useful_work_mwh",
                "fairness",
                "margin_quality",
                "degradation_allocation_quality",
            ],
        )
        w.writeheader()
        for proto, data in results.items():
            w.writerow(
                {
                    "protocol": proto,
                    "joint_violations": data.get("joint_violations", 0),
                    "local_violations": data.get("local_violations", 0),
                    "useful_work_mwh": data.get("useful_work_mwh", 0),
                    "fairness": data.get("fairness", 0),
                    "margin_quality": data.get("margin_quality", 0),
                    "degradation_allocation_quality": data.get(
                        "degradation_allocation_quality", data.get("margin_quality", 0)
                    ),
                }
            )

    # fairness_metrics.csv
    fairness_path = out / "fairness_metrics.csv"
    with open(fairness_path, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "protocol",
                "fairness",
                "margin_quality",
                "degradation_allocation_quality",
                "joint_violations",
                "useful_work_mwh",
            ],
        )
        w.writeheader()
        for proto, data in results.items():
            w.writerow(
                {
                    "protocol": proto,
                    "fairness": data.get("fairness", 0),
                    "margin_quality": data.get("margin_quality", 0),
                    "degradation_allocation_quality": data.get(
                        "degradation_allocation_quality", data.get("margin_quality", 0)
                    ),
                    "joint_violations": data.get("joint_violations", 0),
                    "useful_work_mwh": data.get("useful_work_mwh", 0),
                }
            )

    # Sync publication copy (same schema as protocol_compare)
    pub_dir = Path("reports/publication")
    pub_dir.mkdir(parents=True, exist_ok=True)
    pub_scenario_path = pub_dir / "multi_agent_transformer_scenario.csv"
    with open(pub_scenario_path, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "protocol",
                "joint_violations",
                "local_violations",
                "useful_work_mwh",
                "fairness",
                "margin_quality",
                "degradation_allocation_quality",
            ],
        )
        w.writeheader()
        for proto, data in results.items():
            w.writerow(
                {
                    "protocol": proto,
                    "joint_violations": data.get("joint_violations", 0),
                    "local_violations": data.get("local_violations", 0),
                    "useful_work_mwh": data.get("useful_work_mwh", 0),
                    "fairness": data.get("fairness", 0),
                    "margin_quality": data.get("margin_quality", 0),
                    "degradation_allocation_quality": data.get(
                        "degradation_allocation_quality", data.get("margin_quality", 0)
                    ),
                }
            )

    print("=== Multi-Agent Non-Composition Counterexample ===")
    for proto, s in summary.items():
        print(
            f"  {proto:30s} | violations={s['joint_violations']} "
            f"| useful_work={s['useful_work']:.1f} | fairness={s['fairness']:.4f}"
        )
    print(f"\nSummary → {summary_path}")
    print(f"Protocol compare → {proto_compare_path}")
    print(f"Fairness metrics → {fairness_path}")


if __name__ == "__main__":
    main()
