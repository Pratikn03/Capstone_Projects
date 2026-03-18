#!/usr/bin/env python3
"""Generate Paper 2 runtime horizon trace: per-step tau_t, remaining certified time, triggers.

Outputs: reports/paper2/runtime_horizon_trace.csv

Columns: step, tau_t, remaining_certified_time, expiration_trigger_reason, renewal_trigger_reason
"""
from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "temporal_theorems",
    REPO_ROOT / "src" / "orius" / "dc3s" / "temporal_theorems.py",
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
certificate_validity_horizon = mod.certificate_validity_horizon
certificate_half_life = mod.certificate_half_life
should_renew_certificate = mod.should_renew_certificate
should_expire_certificate = mod.should_expire_certificate

OUT_DIR = REPO_ROOT / "reports" / "paper2"
RENEWAL_THRESHOLD_STEPS = 5


def main() -> int:
    constraints = {
        "min_soc_mwh": 10.0,
        "max_soc_mwh": 90.0,
        "time_step_hours": 1.0,
        "charge_efficiency": 0.95,
        "discharge_efficiency": 0.95,
    }
    safe_action = {"charge_mw": 5.0, "discharge_mw": 0.0}
    initial_soc = 50.0
    margin_mwh = 2.0
    interval_lower = initial_soc - margin_mwh
    interval_upper = initial_soc + margin_mwh
    sigma_d = 1.0

    # Compute tau_t at cert issuance (step 0)
    horizon_result = certificate_validity_horizon(
        interval_lower_mwh=interval_lower,
        interval_upper_mwh=interval_upper,
        safe_action=safe_action,
        constraints=constraints,
        sigma_d=sigma_d,
    )
    tau_t = int(horizon_result["tau_t"])
    half_life_result = certificate_half_life(tau_t=tau_t, decay_rate=0.5)
    half_life_steps = int(half_life_result["half_life_steps"])

    # Simulate N steps (blackout: no renewal, cert consumed step-by-step)
    n_steps = min(48, max(tau_t + 5, 24))
    rows = []

    for step in range(n_steps):
        steps_since_renewal = step
        renew = should_renew_certificate(
            tau_t=tau_t,
            steps_since_renewal=steps_since_renewal,
            renewal_threshold_steps=RENEWAL_THRESHOLD_STEPS,
        )
        expire = should_expire_certificate(
            tau_t=tau_t,
            steps_since_renewal=steps_since_renewal,
        )
        remaining = int(renew["remaining_certified_steps"])
        expiration_reason = expire["expiration_trigger_reason"] or ""
        renewal_reason = renew["renewal_trigger_reason"] or ""

        rows.append({
            "step": step,
            "tau_t": tau_t,
            "remaining_certified_time": remaining,
            "expiration_trigger_reason": expiration_reason,
            "renewal_trigger_reason": renewal_reason,
        })

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "runtime_horizon_trace.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["step", "tau_t", "remaining_certified_time", "expiration_trigger_reason", "renewal_trigger_reason"],
        )
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {out_path} ({len(rows)} steps, tau_t={tau_t}, half_life={half_life_steps})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
