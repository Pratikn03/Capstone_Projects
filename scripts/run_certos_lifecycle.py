#!/usr/bin/env python3
"""CertOS lifecycle demonstration runner (Paper 6).

Simulates a battery dispatch loop with the CertOS runtime. The
validity horizon decays over time, triggering degraded → fallback
transitions and eventual recovery. Outputs CSV + summary JSON.

Usage:
    python scripts/run_certos_lifecycle.py [--steps 96] [--out reports/certos]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from orius.certos.runtime import CertOSConfig, CertOSRuntime


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=96)
    parser.add_argument("--out", default="reports/certos")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cfg = CertOSConfig(
        soc_min_mwh=20.0,
        soc_max_mwh=180.0,
        capacity_mwh=200.0,
        sigma_d=5.0,
        degraded_threshold=4,
    )
    rt = CertOSRuntime(config=cfg)

    rng = np.random.default_rng(42)
    soc = 100.0  # starting SOC
    rows = []

    for t in range(args.steps):
        # Simulate decaying validity horizon with noise
        base_h = max(0, 20 - t // 4 + int(rng.integers(-2, 3)))
        # At step 40-55 simulate a blackout → H_t drops to 0
        if 40 <= t <= 55:
            base_h = 0

        proposed = {"charge_mw": 0.0, "discharge_mw": 50.0}
        safe = {"charge_mw": 0.0, "discharge_mw": min(50.0, max(0.0, 50.0 * base_h / 20))}

        state = rt.validate_and_step(
            observed_soc_mwh=soc,
            proposed_action=proposed,
            safe_action=safe,
            validity_horizon=base_h,
        )

        # Simulate SOC physics
        discharge = state.safe_action.get("discharge_mw", 0)
        soc -= discharge * 0.25 / 0.95  # dt=0.25h, eta=0.95
        soc = max(0, soc)

        invariant_ok = rt.check_invariants(state)

        rows.append({
            "step": t,
            "validity_horizon": state.validity_horizon,
            "status": state.status,
            "discharge_mw": discharge,
            "soc_mwh": round(soc, 2),
            "fallback": state.fallback_active,
            "invariant_violations": ",".join(invariant_ok) if invariant_ok else "none",
        })

    # Write CSV
    csv_path = out / "certos_lifecycle.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Summary
    n_valid = sum(1 for r in rows if r["status"] == "valid")
    n_degraded = sum(1 for r in rows if r["status"] == "degraded")
    n_fallback = sum(1 for r in rows if r["status"] == "fallback")
    summary = {
        "total_steps": args.steps,
        "valid_steps": n_valid,
        "degraded_steps": n_degraded,
        "fallback_steps": n_fallback,
        "interventions": rt.intervention_count,
        "audit_entries": len(rt.audit_log),
    }
    summary_path = out / "certos_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("=== CertOS Lifecycle Demo ===")
    print(f"  Valid steps    : {n_valid}")
    print(f"  Degraded steps : {n_degraded}")
    print(f"  Fallback steps : {n_fallback}")
    print(f"  Interventions  : {rt.intervention_count}")
    print(f"  Audit entries  : {len(rt.audit_log)}")
    print(f"\n  CSV     → {csv_path}")
    print(f"  Summary → {summary_path}")


if __name__ == "__main__":
    main()
