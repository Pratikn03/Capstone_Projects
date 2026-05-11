from orius.dc3s.temporal_theorems import (
    certificate_validity_horizon,
    should_expire_certificate,
    should_renew_certificate,
)

CONSTRAINTS = {
    "min_soc_mwh": 20.0,
    "max_soc_mwh": 180.0,
    "time_step_hours": 1.0,
    "charge_efficiency": 1.0,
    "discharge_efficiency": 1.0,
}
HOLD_ACTION = {"charge_mw": 0.0, "discharge_mw": 0.0}


def test_certificate_horizon_positive_under_clean_telemetry():
    result = certificate_validity_horizon(
        interval_lower_mwh=90.0,
        interval_upper_mwh=100.0,
        safe_action=HOLD_ACTION,
        constraints=CONSTRAINTS,
        sigma_d=0.1,
        max_steps=24,
    )

    assert result["tau_t"] == 24
    assert result["tube_lower_mwh"] >= CONSTRAINTS["min_soc_mwh"]
    assert result["tube_upper_mwh"] <= CONSTRAINTS["max_soc_mwh"]


def test_certificate_horizon_shrinks_under_degradation():
    clean = certificate_validity_horizon(
        interval_lower_mwh=90.0,
        interval_upper_mwh=100.0,
        safe_action=HOLD_ACTION,
        constraints=CONSTRAINTS,
        sigma_d=0.1,
        max_steps=100,
    )
    degraded = certificate_validity_horizon(
        interval_lower_mwh=90.0,
        interval_upper_mwh=100.0,
        safe_action=HOLD_ACTION,
        constraints=CONSTRAINTS,
        sigma_d=10.0,
        max_steps=100,
    )

    assert degraded["tau_t"] < clean["tau_t"]


def test_certificate_horizon_zero_when_current_tube_unsafe():
    result = certificate_validity_horizon(
        interval_lower_mwh=10.0,
        interval_upper_mwh=30.0,
        safe_action=HOLD_ACTION,
        constraints=CONSTRAINTS,
        sigma_d=0.1,
        max_steps=12,
    )

    assert result["tau_t"] == 0


def test_certificate_expiry_and_renewal_gates_fail_closed():
    renew = should_renew_certificate(tau_t=8, steps_since_renewal=4, renewal_threshold_steps=5)
    expire = should_expire_certificate(tau_t=8, steps_since_renewal=8)

    assert renew["should_renew"] is True
    assert expire["should_expire"] is True
    assert expire["remaining_certified_steps"] == 0
