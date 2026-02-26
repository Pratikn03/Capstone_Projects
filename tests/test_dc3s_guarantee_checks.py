"""Unit tests for deterministic DC3S guarantee checks."""
from __future__ import annotations

from gridpulse.dc3s.guarantee_checks import (
    check_no_simultaneous_charge_discharge,
    check_power_bounds,
    check_soc_invariance,
    evaluate_guarantee_checks,
)


def _constraints() -> dict[str, float]:
    return {
        "min_soc_mwh": 10.0,
        "max_soc_mwh": 90.0,
        "max_power_mw": 20.0,
        "max_charge_mw": 20.0,
        "max_discharge_mw": 20.0,
        "time_step_hours": 1.0,
        "charge_efficiency": 0.95,
        "discharge_efficiency": 0.95,
    }


def test_guarantee_checks_pass_for_safe_action() -> None:
    constraints = _constraints()
    action = {"charge_mw": 5.0, "discharge_mw": 0.0}

    assert check_no_simultaneous_charge_discharge(action)
    assert check_power_bounds(action, constraints)
    assert check_soc_invariance(40.0, action, constraints)

    ok, reasons, _ = evaluate_guarantee_checks(current_soc=40.0, action=action, constraints=constraints)
    assert ok is True
    assert reasons == []


def test_guarantee_checks_fail_for_unsafe_action() -> None:
    constraints = _constraints()
    action = {"charge_mw": 25.0, "discharge_mw": 25.0}
    ok, reasons, _ = evaluate_guarantee_checks(current_soc=89.0, action=action, constraints=constraints)

    assert ok is False
    assert "simultaneous_charge_discharge" in reasons
    assert "power_bounds" in reasons
