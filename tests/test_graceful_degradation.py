"""Tests for Paper 3 graceful degradation planner."""
from __future__ import annotations

import pytest

from orius.dc3s.graceful import plan_graceful_degradation, optimized_graceful, compare_policies


def test_graceful_degradation_planner_no_fallback():
    state = {"fallback_required": False}
    plan = plan_graceful_degradation(state, {}, {}, "ramp_down", 10)
    assert not plan["actions"]
    assert "not required" in plan["reason"]


def test_graceful_degradation_planner_hard_shutdown():
    state = {"fallback_required": True}
    plan = plan_graceful_degradation(state, {}, {}, "hard_shutdown", 10)
    assert len(plan["actions"]) == 1
    assert plan["actions"][0]["charge_mw"] == 0.0
    assert plan["actions"][0]["discharge_mw"] == 0.0


def test_graceful_degradation_planner_optimized_constructive():
    """Optimized mode produces tapered actions over remaining horizon."""
    state = {
        "fallback_required": True,
        "last_action": {"charge_mw": 0.0, "discharge_mw": 0.0},
        "current_soc_mwh": 5000.0,
        "constraints": {
            "min_soc_mwh": 0.0,
            "max_soc_mwh": 10000.0,
            "time_step_hours": 1.0,
            "charge_efficiency": 0.95,
            "discharge_efficiency": 0.95,
        },
    }
    plan = plan_graceful_degradation(state, {}, {"useful_work_weight": 1.0}, "optimized", 8)
    assert len(plan["actions"]) == 8
    assert "Optimized" in plan["reason"]


def test_optimized_graceful_consumes_horizon():
    """optimized_graceful returns horizon_steps actions, tapered to zero."""
    c = {
        "min_soc_mwh": 0.0,
        "max_soc_mwh": 100.0,
        "time_step_hours": 1.0,
        "charge_efficiency": 0.95,
        "discharge_efficiency": 0.95,
    }
    acts = optimized_graceful({"charge_mw": 0, "discharge_mw": 20.0}, 5, 50.0, c)
    assert len(acts) == 5
    assert acts[0]["discharge_mw"] > acts[-1]["discharge_mw"]


def test_compare_policies_four_policies():
    """compare_policies returns all four policies."""
    c = {"min_soc_mwh": 0, "max_soc_mwh": 100, "time_step_hours": 1, "charge_efficiency": 0.95, "discharge_efficiency": 0.95}
    r = compare_policies({"charge_mw": 0, "discharge_mw": 0.0}, 5, 50.0, c)
    assert set(r.keys()) == {"blind_persistence", "immediate_shutdown", "simple_ramp_down", "optimized_graceful"}