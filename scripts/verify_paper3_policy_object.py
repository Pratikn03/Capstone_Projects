#!/usr/bin/env python3
"""Verify Paper 3 graceful degradation policy object (Step 3.1).

Runs without full orius deps (direct import of graceful.py).
Outputs to stdout; no artifact. Use for evidence-first verification.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("graceful", REPO / "src" / "orius" / "dc3s" / "graceful.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def main() -> int:
    constraints = {
        "min_soc_mwh": 10.0,
        "max_soc_mwh": 90.0,
        "time_step_hours": 1.0,
        "charge_efficiency": 0.95,
        "discharge_efficiency": 0.95,
    }
    last_action = {"charge_mw": 5.0, "discharge_mw": 0.0}
    state = {
        "fallback_required": True,
        "last_action": last_action,
        "current_soc_mwh": 50.0,
        "constraints": constraints,
    }

    # 1. plan_graceful_degradation consumes remaining_horizon
    plan = mod.plan_graceful_degradation(state, {}, {"useful_work_weight": 0.8}, "optimized", 10)
    assert len(plan["actions"]) == 10, f"Expected 10 actions, got {len(plan['actions'])}"
    assert plan["actions"][-1]["charge_mw"] < plan["actions"][0]["charge_mw"], "Taper should decrease"

    # 2. optimized_graceful constructive
    acts = list(mod.optimized_graceful(last_action, 5, 50.0, constraints, sigma_d=50.0, utility_weight=1.0))
    assert len(acts) == 5
    assert acts[-1]["charge_mw"] < acts[0]["charge_mw"]

    # 3. compare_policies four policies
    result = mod.compare_policies(last_action, 10, 50.0, constraints)
    assert set(result.keys()) == {"blind_persistence", "immediate_shutdown", "simple_ramp_down", "optimized_graceful"}

    out = REPO / "reports" / "paper3"
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "plan_actions": len(plan["actions"]),
        "plan_reason": plan["reason"],
        "policies": list(result.keys()),
        "gdq_by_policy": {p: r["gdq"] for p, r in result.items()},
    }
    (out / "policy_object_verify.json").write_text(json.dumps(summary, indent=2))
    print("Paper 3 policy object: OK")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
