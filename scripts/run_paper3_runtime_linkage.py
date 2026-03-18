#!/usr/bin/env python3
"""Paper 3 Step 3.3: Runtime linkage demo.

Proves: certificate horizon (tau_t) → planner (optimized_graceful) → tapered actions.
Output: reports/paper3/runtime_linkage_trace.json

Uses importlib to load temporal_theorems and graceful directly, avoiding the full
orius package (which pulls numpy via dc3s/quality/ftit).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _resolve_repo_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else REPO / path


def _load_temporal_theorems():
    spec = importlib.util.spec_from_file_location(
        "temporal_theorems",
        REPO / "src" / "orius" / "dc3s" / "temporal_theorems.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_graceful():
    spec = importlib.util.spec_from_file_location(
        "graceful",
        REPO / "src" / "orius" / "dc3s" / "graceful.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    parser = argparse.ArgumentParser(description="Paper 3 runtime linkage demo")
    parser.add_argument("--out", default="reports/paper3", help="Output directory")
    args = parser.parse_args()

    out_dir = _resolve_repo_path(args.out)

    temporal = _load_temporal_theorems()
    graceful = _load_graceful()
    certificate_validity_horizon = temporal.certificate_validity_horizon
    should_expire_certificate = temporal.should_expire_certificate
    optimized_graceful = graceful.optimized_graceful

    out_dir.mkdir(parents=True, exist_ok=True)

    constraints = {
        "min_soc_mwh": 10.0,
        "max_soc_mwh": 90.0,
        "time_step_hours": 1.0,
        "charge_efficiency": 0.95,
        "discharge_efficiency": 0.95,
    }
    last_action = {"charge_mw": 5.0, "discharge_mw": 0.0}
    soc_mwh = 50.0
    sigma_d = 0.5
    margin = 10.0

    # 1. Compute certificate horizon (tau_t) from certificate_validity_horizon
    interval_lower = soc_mwh - margin
    interval_upper = soc_mwh + margin
    result = certificate_validity_horizon(
        interval_lower_mwh=interval_lower,
        interval_upper_mwh=interval_upper,
        safe_action=last_action,
        constraints=constraints,
        sigma_d=sigma_d,
    )
    tau_t = result["tau_t"]

    # 2. Planner uses tau_t (certificate horizon), not fixed timer
    actions = list(optimized_graceful(
        last_action=last_action,
        horizon_steps=tau_t,
        soc_mwh=soc_mwh,
        constraints=constraints,
        sigma_d=sigma_d,
    ))

    # 3. Fallback termination: should_expire when remaining <= 0
    expire_result = should_expire_certificate(tau_t=tau_t, steps_since_renewal=tau_t)

    trace = {
        "certificate_horizon_tau_t": tau_t,
        "planner_horizon_steps": len(actions),
        "planner_uses_certificate_horizon": tau_t == len(actions),
        "actions_taper_sample": [
            {"step": i, "discharge_mw": round(a["discharge_mw"], 4), "charge_mw": round(a["charge_mw"], 4)}
            for i, a in enumerate(actions[:3])
        ]
        + ([{"step": len(actions) - 1, "discharge_mw": round(actions[-1]["discharge_mw"], 4), "charge_mw": round(actions[-1]["charge_mw"], 4)}] if len(actions) > 3 else []),
        "taper_verified": (
            (actions[0]["discharge_mw"] > actions[-1]["discharge_mw"] or actions[0]["charge_mw"] > actions[-1]["charge_mw"])
            if len(actions) > 1 else True
        ),
        "fallback_termination": {
            "should_expire_after_tau_t_steps": expire_result["should_expire"],
            "remaining_certified_steps": expire_result["remaining_certified_steps"],
        },
    }

    path = out_dir / "runtime_linkage_trace.json"
    path.write_text(json.dumps(trace, indent=2))
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
