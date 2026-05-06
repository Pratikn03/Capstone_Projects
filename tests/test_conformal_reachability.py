"""Tests for Paper 2 conformal reachability propagation."""

from __future__ import annotations

from orius.dc3s.reachability import (
    compute_expiration_bound,
    compute_validity_horizon_from_reachability,
    propagate_reachability_set,
)


def test_propagate_reachability_set() -> None:
    tube = propagate_reachability_set(
        interval_lower_mwh=40.0,
        interval_upper_mwh=60.0,
        safe_action={"charge_mw": 0.0, "discharge_mw": 0.0},
        horizon_steps=10,
        sigma_d=1.0,
    )
    assert "lower_mwh" in tube
    assert "upper_mwh" in tube
    assert "radius_mwh" in tube
    assert tube["radius_mwh"] >= 0


def test_compute_validity_horizon_from_reachability() -> None:
    result = compute_validity_horizon_from_reachability(
        interval_lower_mwh=40.0,
        interval_upper_mwh=60.0,
        safe_action={"charge_mw": 0.0, "discharge_mw": 0.0},
        constraints={
            "min_soc_mwh": 10.0,
            "max_soc_mwh": 90.0,
            "time_step_hours": 1.0,
            "charge_efficiency": 0.95,
            "discharge_efficiency": 0.95,
        },
        sigma_d=1.0,
        max_steps=100,
    )
    assert result["tau_t"] >= 0


def test_compute_expiration_bound() -> None:
    result = compute_expiration_bound(
        interval_lower_mwh=45.0,
        interval_upper_mwh=55.0,
        soc_min_mwh=10.0,
        soc_max_mwh=90.0,
        sigma_d=1.0,
    )
    assert "tau_expire_lb" in result
    assert "delta_bnd_mwh" in result
    assert result["tau_expire_lb"] >= 0
