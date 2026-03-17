"""Tests for Paper 3 graceful degradation planner."""
from __future__ import annotations

import pytest

from orius.dc3s.graceful import plan_graceful_degradation


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