#!/usr/bin/env python3
"""Run vehicle prototype benchmark (ORIUS extension).

Outputs to reports/vehicles_prototype/ — isolated from locked battery artifacts.
Prototype only; not part of paper claims.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from orius.adapters.vehicle import VehicleDomainAdapter
from orius.vehicles.plant import VehiclePlant
from orius.vehicles.vehicle_runner import compute_vehicle_metrics, run_vehicle_episode


def main() -> None:
    out_dir = REPO_ROOT / "reports" / "vehicles_prototype"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = {"expected_cadence_s": 1.0, "reliability": {"min_w": 0.05}}
    adapter = VehicleDomainAdapter(cfg)
    plant = VehiclePlant(dt_s=0.25, speed_limit_mps=30.0)

    results = run_vehicle_episode(
        adapter=adapter,
        plant=plant,
        horizon=48,
        seed=42,
        fault_inject=False,
    )
    metrics = compute_vehicle_metrics(results)

    csv_path = out_dir / "vehicle_episode_log.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step", "speed_mps", "position_m", "violated", "intervened", "w_t"])
        w.writeheader()
        for r in results:
            w.writerow(
                {
                    "step": r.step,
                    "speed_mps": r.true_state.get("speed_mps", 0),
                    "position_m": r.true_state.get("position_m", 0),
                    "violated": r.violated,
                    "intervened": r.intervened,
                    "w_t": r.w_t,
                }
            )

    metrics_path = out_dir / "vehicle_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("Vehicle prototype benchmark complete.")
    print(f"  Violations: {metrics['speed_limit_violations_pct']:.2f}%")
    print(f"  Interventions: {metrics['intervention_rate_pct']:.2f}%")
    print(f"  Log -> {csv_path}")
    print(f"  Metrics -> {metrics_path}")


if __name__ == "__main__":
    main()
