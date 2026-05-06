#!/usr/bin/env python3
"""Paper 2 Step 2.1: Runtime horizon trace.

Produces reports/paper2/runtime_horizon_trace.csv with per-step:
- step
- tau_t (current horizon)
- remaining_certified_time
- expiration_trigger_reason
- renewal_trigger_reason

Uses: forward_tube, certificate_validity_horizon, certificate_half_life,
      should_renew_certificate, should_expire_certificate.

Claim: ch28, theorem_to_evidence_map T5-T6, app_m.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from orius.dc3s.temporal_theorems import (
    certificate_half_life,
    certificate_validity_horizon,
    should_expire_certificate,
    should_renew_certificate,
)


def main() -> int:
    out = REPO / "reports" / "paper2"
    out.mkdir(parents=True, exist_ok=True)

    constraints = {
        "min_soc_mwh": 20.0,
        "max_soc_mwh": 180.0,
        "time_step_hours": 1.0,
        "charge_efficiency": 0.95,
        "discharge_efficiency": 0.95,
    }
    safe_action = {"charge_mw": 0.0, "discharge_mw": 50.0}
    sigma_d = 2.0
    renewal_threshold = 5
    max_steps = 24

    # Simulate decaying interval (blackout-style: interval widens then shrinks)
    initial_soc = 100.0
    margin = 10.0
    interval_lower = initial_soc - margin
    interval_upper = initial_soc + margin

    rows = []
    steps_since_renewal = 0

    for step in range(max_steps):
        # Decay interval width over time (simulate forecast uncertainty growth)
        decay = 1.0 + step * 0.05
        il = interval_lower - decay
        iu = interval_upper + decay
        il = max(constraints["min_soc_mwh"] + 1.0, il)
        iu = min(constraints["max_soc_mwh"] - 1.0, iu)

        result = certificate_validity_horizon(
            interval_lower_mwh=il,
            interval_upper_mwh=iu,
            safe_action=safe_action,
            constraints=constraints,
            sigma_d=sigma_d,
        )
        tau_t = result["tau_t"]
        half_life_result = certificate_half_life(tau_t=tau_t, decay_rate=0.5)

        renew = should_renew_certificate(
            tau_t=tau_t,
            steps_since_renewal=steps_since_renewal,
            renewal_threshold_steps=renewal_threshold,
        )
        expire = should_expire_certificate(
            tau_t=tau_t,
            steps_since_renewal=steps_since_renewal,
        )

        remaining = renew["remaining_certified_steps"]
        exp_reason = expire["expiration_trigger_reason"] or ""
        ren_reason = renew["renewal_trigger_reason"] or ""

        rows.append(
            {
                "step": step,
                "tau_t": tau_t,
                "remaining_certified_time": remaining,
                "expiration_trigger_reason": exp_reason,
                "renewal_trigger_reason": ren_reason,
                "half_life_steps": half_life_result["half_life_steps"],
            }
        )

        steps_since_renewal += 1

    path = out / "runtime_horizon_trace.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
