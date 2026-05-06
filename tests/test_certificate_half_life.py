"""Tests for Paper 2 certificate half-life and validity horizon."""

from __future__ import annotations

from orius.dc3s.half_life import (
    compute_certificate_state,
    compute_half_life_from_horizon,
    compute_validity_horizon,
)


def test_compute_validity_horizon_interior_soc() -> None:
    result = compute_validity_horizon(
        observed_state={"current_soc_mwh": 50.0},
        quality_score=0.9,
        safety_margin_mwh=5.0,
        constraints={
            "min_soc_mwh": 10.0,
            "max_soc_mwh": 90.0,
            "time_step_hours": 1.0,
            "charge_efficiency": 0.95,
            "discharge_efficiency": 0.95,
        },
        sigma_d=1.0,
    )
    assert result["tau_t"] >= 0
    assert "soc_min_mwh" in result
    assert "soc_max_mwh" in result


def test_compute_half_life_from_horizon() -> None:
    result = compute_half_life_from_horizon(tau_t=100, decay_rate=0.5)
    assert result["half_life_steps"] == 100
    assert result["tau_t"] == 100

    result2 = compute_half_life_from_horizon(tau_t=100, decay_rate=0.25)
    assert result2["half_life_steps"] < 100


def test_compute_certificate_state() -> None:
    state = compute_certificate_state(
        observed_state={"current_soc_mwh": 50.0},
        quality_score=0.9,
        safety_margin_mwh=5.0,
        constraints={
            "min_soc_mwh": 10.0,
            "max_soc_mwh": 90.0,
            "time_step_hours": 1.0,
            "charge_efficiency": 0.95,
            "discharge_efficiency": 0.95,
        },
        sigma_d=1.0,
        current_step=10,
    )
    assert state["status"] in ("valid", "expiring", "expired")
    assert "H_t" in state
    assert "half_life" in state
    assert "expires_at_step" in state
    assert "renewal_ready" in state
    assert "fallback_required" in state
    assert state["expires_at_step"] == 10 + state["H_t"]
