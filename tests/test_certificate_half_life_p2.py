"""Tests for certificate half-life and runtime validity-horizon engine (Paper 2)."""
from __future__ import annotations

import pytest
from orius.dc3s.half_life import (
    compute_certificate_state,
    check_renewal_trigger,
    time_to_expiration,
)


_CONSTRAINTS = {
    "min_soc_mwh": 0.0,
    "max_soc_mwh": 10000.0,
    "capacity_mwh": 10000.0,
    "time_step_hours": 1.0,
    "charge_efficiency": 0.95,
    "discharge_efficiency": 0.95,
}


class TestComputeCertificateState:
    def test_valid_state_center_soc(self):
        """Mid-range SOC with full quality -> valid."""
        state = compute_certificate_state(
            observed_soc_mwh=5000.0,
            quality_score=1.0,
            safety_margin_mwh=100.0,
            sigma_d=50.0,
            constraints=_CONSTRAINTS,
        )
        assert state["status"] == "valid"
        assert state["H_t"] > 0
        assert state["half_life"] > 0
        assert state["fallback_required"] is False
        assert state["renewal_ready"] is True

    def test_expired_state_extreme_soc(self):
        """SOC near boundary with high sigma -> expired quickly."""
        state = compute_certificate_state(
            observed_soc_mwh=9999.0,
            quality_score=0.8,
            safety_margin_mwh=500.0,
            sigma_d=500.0,
            constraints=_CONSTRAINTS,
        )
        # Near max SOC with large sigma, horizon should be very short or 0
        assert state["H_t"] <= 2
        assert state["status"] in ("expired", "degraded")

    def test_low_quality_forces_fallback(self):
        """w_t < 0.05 -> fallback_required status."""
        state = compute_certificate_state(
            observed_soc_mwh=5000.0,
            quality_score=0.02,
            safety_margin_mwh=100.0,
            sigma_d=50.0,
            constraints=_CONSTRAINTS,
        )
        assert state["status"] == "fallback_required"
        assert state["fallback_required"] is True

    def test_confidence_decreases_with_step(self):
        """Confidence should decrease as current_step increases."""
        c0 = compute_certificate_state(
            observed_soc_mwh=5000.0, quality_score=1.0,
            safety_margin_mwh=100.0, sigma_d=50.0,
            constraints=_CONSTRAINTS, current_step=0,
        )["confidence"]
        c5 = compute_certificate_state(
            observed_soc_mwh=5000.0, quality_score=1.0,
            safety_margin_mwh=100.0, sigma_d=50.0,
            constraints=_CONSTRAINTS, current_step=5,
        )["confidence"]
        assert c0 >= c5

    def test_zero_sigma_infinite_horizon(self):
        """sigma_d=0 -> no disturbance growth -> very large H_t."""
        state = compute_certificate_state(
            observed_soc_mwh=5000.0,
            quality_score=1.0,
            safety_margin_mwh=100.0,
            sigma_d=0.0,
            constraints=_CONSTRAINTS,
        )
        assert state["H_t"] >= 4096  # capped at max_steps
        assert state["status"] == "valid"

    def test_expires_at_step_calculation(self):
        state = compute_certificate_state(
            observed_soc_mwh=5000.0, quality_score=1.0,
            safety_margin_mwh=100.0, sigma_d=50.0,
            constraints=_CONSTRAINTS, current_step=10,
        )
        assert state["expires_at_step"] == 10 + state["H_t"]


class TestRenewalTrigger:
    def test_renewal_when_degraded(self):
        cert_state = {
            "status": "degraded",
            "H_t": 2,
        }
        result = check_renewal_trigger(
            certificate_state=cert_state,
            new_quality_score=0.8,
        )
        assert result["should_renew"] is True

    def test_no_renewal_when_valid(self):
        cert_state = {
            "status": "valid",
            "H_t": 100,
        }
        result = check_renewal_trigger(
            certificate_state=cert_state,
            new_quality_score=0.9,
        )
        assert result["should_renew"] is False

    def test_no_renewal_low_quality(self):
        cert_state = {
            "status": "degraded",
            "H_t": 1,
        }
        result = check_renewal_trigger(
            certificate_state=cert_state,
            new_quality_score=0.2,
        )
        assert result["should_renew"] is False


class TestTimeToExpiration:
    def test_remaining_positive(self):
        r = time_to_expiration(H_t=48, elapsed_since_issue=10, dt_hours=1.0)
        assert r["remaining_steps"] == 38
        assert r["remaining_hours"] == 38.0
        assert r["expired"] is False

    def test_expired(self):
        r = time_to_expiration(H_t=10, elapsed_since_issue=15)
        assert r["remaining_steps"] == 0
        assert r["expired"] is True
