import pytest

from orius.dc3s.temporal_theorems import (
    certificate_expiration_bound,
    certificate_validity_horizon,
    certify_fallback_existence,
    evaluate_graceful_degradation_dominance,
    forward_tube,
    zero_dispatch_fallback,
)


def test_forward_tube_matches_battery_action_and_drift_radius():
    tube = forward_tube(
        interval_lower_mwh=45.0,
        interval_upper_mwh=55.0,
        safe_action={"charge_mw": 10.0, "discharge_mw": 0.0},
        horizon_steps=4,
        sigma_d=2.0,
        dt_hours=1.0,
        charge_efficiency=1.0,
        discharge_efficiency=1.0,
    )
    assert tube["net_action_delta_mwh"] == pytest.approx(40.0)
    assert tube["radius_mwh"] == pytest.approx(4.0)
    assert tube["lower_mwh"] == pytest.approx(81.0)
    assert tube["upper_mwh"] == pytest.approx(99.0)


def test_certificate_validity_horizon_matches_expiration_bound_for_zero_action():
    constraints = {"min_soc_mwh": 0.0, "max_soc_mwh": 100.0, "time_step_hours": 1.0}
    horizon = certificate_validity_horizon(
        interval_lower_mwh=45.0,
        interval_upper_mwh=55.0,
        safe_action=zero_dispatch_fallback(),
        constraints=constraints,
        sigma_d=5.0,
        max_steps=200,
    )
    bound = certificate_expiration_bound(
        interval_lower_mwh=45.0,
        interval_upper_mwh=55.0,
        soc_min_mwh=0.0,
        soc_max_mwh=100.0,
        sigma_d=5.0,
    )
    assert bound["tau_expire_lb"] == 81
    assert horizon["tau_t"] == 81


def test_certify_fallback_existence_passes_for_interior_soc():
    result = certify_fallback_existence(
        current_soc_mwh=50.0,
        constraints={"min_soc_mwh": 10.0, "max_soc_mwh": 90.0},
        model_error_mwh=5.0,
    )
    assert result["passed"] is True
    assert result["fallback_action"] == {"charge_mw": 0.0, "discharge_mw": 0.0}


def test_certify_fallback_existence_fails_near_boundary():
    result = certify_fallback_existence(
        current_soc_mwh=11.0,
        constraints={"min_soc_mwh": 10.0, "max_soc_mwh": 90.0},
        model_error_mwh=2.0,
    )
    assert result["passed"] is False


def test_graceful_degradation_dominance_detects_strict_domination():
    result = evaluate_graceful_degradation_dominance(
        graceful_violations=[0, 0, 0, 0],
        uncontrolled_violations=[0, 1, 0, 1],
    )
    assert result["dominates"] is True
    assert result["strict_if_uncontrolled_violates"] is True
    assert result["graceful_violation_count"] == pytest.approx(0.0)
    assert result["uncontrolled_violation_count"] == pytest.approx(2.0)


def test_graceful_degradation_dominance_detects_failure():
    result = evaluate_graceful_degradation_dominance(
        graceful_violations=[1, 0],
        uncontrolled_violations=[0, 0],
    )
    assert result["dominates"] is False
