#!/usr/bin/env python3
"""CertOS toy second-domain demo for runtime-governance portability.

Runs CertOSRuntime with vehicle-shaped actions to demonstrate
CertOS works with a peer domain (vehicle). Outputs certos_vehicle_toy.json.

Usage:
    python3 scripts/run_certos_vehicle_toy.py [--out reports/certos]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from orius.certos.runtime import CertOSConfig, CertOSRuntime


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="reports/certos")
    parser.add_argument("--steps", type=int, default=24)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cfg = CertOSConfig(degraded_threshold=4)
    rt = CertOSRuntime(config=cfg)

    n_fallback = 0
    for t in range(args.steps):
        # Simulate validity horizon: blackout at steps 8-12
        h_t = 0 if 8 <= t <= 12 else max(1, 10 - t // 3)
        proposed = {"acceleration_mps2": 1.0}
        safe = {"acceleration_mps2": 0.8} if h_t > 0 else {"acceleration_mps2": 0.0}
        state = rt.validate_and_step(
            observed_soc_mwh=0.0,  # unused for vehicle
            proposed_action=proposed,
            safe_action=safe,
            validity_horizon=h_t,
        )
        if state.fallback_active:
            n_fallback += 1

    result = {
        "domain": "vehicle",
        "steps": args.steps,
        "fallback_steps": n_fallback,
        "audit_entries": len(rt.audit_log),
        "interventions": rt.intervention_count,
    }
    path = out / "certos_vehicle_toy.json"
    with open(path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"CertOS vehicle toy -> {path}")
    print(f"  domain={result['domain']} steps={result['steps']} fallback={result['fallback_steps']}")


if __name__ == "__main__":
    main()
