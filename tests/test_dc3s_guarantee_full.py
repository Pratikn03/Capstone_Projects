"""Comprehensive tests for DC3S guarantee checks and safety filter theory."""
from __future__ import annotations

import pytest

from orius.dc3s.guarantee_checks import (
    check_no_simultaneous_charge_discharge,
    check_power_bounds,
    check_soc_invariance,
    evaluate_guarantee_checks,
    next_soc,
)
from orius.dc3s.safety_filter_theory import (
    check_tightened_soc_invariance,
    reliability_error_bound,
    safety_filter_projection_summary,
    tightened_soc_bounds,
)


def _constraints(**overrides):
    base = {
        "min_soc_mwh": 10.0,
        "max_soc_mwh": 90.0,
        "capacity_mwh": 100.0,
        "max_power_mw": 50.0,
        "max_charge_mw": 50.0,
        "max_discharge_mw": 50.0,
        "time_step_hours": 1.0,
        "charge_efficiency": 0.95,
        "discharge_efficiency": 0.95,
    }
    base.update(overrides)
    return base


class TestNoSimultaneous:
    def test_charge_only(self):
        assert check_no_simultaneous_charge_discharge({"charge_mw": 5.0, "discharge_mw": 0.0})

    def test_discharge_only(self):
        assert check_no_simultaneous_charge_discharge({"charge_mw": 0.0, "discharge_mw": 5.0})

    def test_both_active_fails(self):
        assert not check_no_simultaneous_charge_discharge({"charge_mw": 5.0, "discharge_mw": 5.0})

    def test_both_zero(self):
        assert check_no_simultaneous_charge_discharge({"charge_mw": 0.0, "discharge_mw": 0.0})

    def test_tiny_values_within_eps(self):
        assert check_no_simultaneous_charge_discharge({"charge_mw": 1e-10, "discharge_mw": 1e-10})


class TestPowerBounds:
    def test_within_limits(self):
        assert check_power_bounds({"charge_mw": 10.0, "discharge_mw": 0.0}, _constraints())

    def test_charge_exceeds(self):
        assert not check_power_bounds({"charge_mw": 60.0, "discharge_mw": 0.0}, _constraints())

    def test_discharge_exceeds(self):
        assert not check_power_bounds({"charge_mw": 0.0, "discharge_mw": 60.0}, _constraints())

    def test_at_limit(self):
        assert check_power_bounds({"charge_mw": 50.0, "discharge_mw": 0.0}, _constraints())


class TestNextSoc:
    def test_charge_increases(self):
        soc_val = next_soc(current_soc=50.0, action={"charge_mw": 10.0, "discharge_mw": 0.0},
                            dt_hours=1.0, charge_efficiency=0.95, discharge_efficiency=0.95)
        assert soc_val == pytest.approx(50.0 + 0.95 * 10.0)

    def test_discharge_decreases(self):
        soc_val = next_soc(current_soc=50.0, action={"charge_mw": 0.0, "discharge_mw": 10.0},
                            dt_hours=1.0, charge_efficiency=0.95, discharge_efficiency=0.95)
        assert soc_val == pytest.approx(50.0 - 10.0 / 0.95)

    def test_no_action_preserves(self):
        soc_val = next_soc(current_soc=50.0, action={"charge_mw": 0.0, "discharge_mw": 0.0},
                            dt_hours=1.0, charge_efficiency=0.95, discharge_efficiency=0.95)
        assert soc_val == pytest.approx(50.0)

    def test_dt_hours_scaling(self):
        soc_val = next_soc(current_soc=50.0, action={"charge_mw": 10.0, "discharge_mw": 0.0},
                            dt_hours=0.5, charge_efficiency=1.0, discharge_efficiency=1.0)
        assert soc_val == pytest.approx(50.0 + 5.0)


class TestSocInvariance:
    def test_safe_action_passes(self):
        assert check_soc_invariance(50.0, {"charge_mw": 5.0, "discharge_mw": 0.0}, _constraints())

    def test_exceed_max_fails(self):
        assert not check_soc_invariance(
            89.0, {"charge_mw": 50.0, "discharge_mw": 0.0},
            _constraints(charge_efficiency=1.0),
        )

    def test_below_min_fails(self):
        assert not check_soc_invariance(
            11.0, {"charge_mw": 0.0, "discharge_mw": 50.0},
            _constraints(discharge_efficiency=1.0),
        )

    def test_boundary_ok(self):
        assert check_soc_invariance(
            50.0, {"charge_mw": 0.0, "discharge_mw": 0.0}, _constraints()
        )


class TestEvaluateGuaranteeChecks:
    def test_all_pass(self):
        ok, reasons, proj = evaluate_guarantee_checks(
            current_soc=50.0,
            action={"charge_mw": 5.0, "discharge_mw": 0.0},
            constraints=_constraints(),
        )
        assert ok is True
        assert reasons == []
        assert proj > 50.0

    def test_multiple_failures(self):
        ok, reasons, _ = evaluate_guarantee_checks(
            current_soc=89.0,
            action={"charge_mw": 60.0, "discharge_mw": 60.0},
            constraints=_constraints(charge_efficiency=1.0),
        )
        assert ok is False
        assert "simultaneous_charge_discharge" in reasons
        assert "power_bounds" in reasons

    def test_soc_invariance_failure(self):
        ok, reasons, _ = evaluate_guarantee_checks(
            current_soc=89.0,
            action={"charge_mw": 50.0, "discharge_mw": 0.0},
            constraints=_constraints(charge_efficiency=1.0),
        )
        assert ok is False
        assert "soc_invariance" in reasons


class TestReliabilityErrorBound:
    def test_monotone(self):
        clean = reliability_error_bound(reliability_w=0.95, max_error_mwh=10.0)
        degraded = reliability_error_bound(reliability_w=0.25, max_error_mwh=10.0)
        assert degraded > clean

    def test_zero_at_perfect(self):
        err = reliability_error_bound(reliability_w=1.0, max_error_mwh=10.0, min_error_mwh=0.0)
        assert err == pytest.approx(0.0)

    def test_max_at_zero_w(self):
        err = reliability_error_bound(reliability_w=0.0, max_error_mwh=10.0, min_error_mwh=0.0)
        assert err == pytest.approx(10.0)

    def test_power_parameter(self):
        lin = reliability_error_bound(reliability_w=0.5, max_error_mwh=10.0, power=1.0)
        quad = reliability_error_bound(reliability_w=0.5, max_error_mwh=10.0, power=2.0)
        assert lin != quad

    def test_min_error_floor(self):
        err = reliability_error_bound(reliability_w=1.0, max_error_mwh=10.0, min_error_mwh=2.0)
        assert err == pytest.approx(2.0)


class TestTightenedSocBounds:
    def test_tighter_than_raw(self):
        lo, hi = tightened_soc_bounds(min_soc_mwh=10.0, max_soc_mwh=90.0, error_bound_mwh=5.0)
        assert lo > 10.0
        assert hi < 90.0

    def test_collapse_to_midpoint(self):
        lo, hi = tightened_soc_bounds(min_soc_mwh=10.0, max_soc_mwh=90.0, error_bound_mwh=50.0)
        assert lo == pytest.approx(hi)
        assert lo == pytest.approx(50.0)

    def test_zero_error_unchanged(self):
        lo, hi = tightened_soc_bounds(min_soc_mwh=10.0, max_soc_mwh=90.0, error_bound_mwh=0.0)
        assert lo == pytest.approx(10.0)
        assert hi == pytest.approx(90.0)


class TestTightenedSocInvariance:
    def test_observed_safe_implies_true_safe(self):
        res = check_tightened_soc_invariance(
            current_soc_obs=50.0,
            action={"charge_mw": 5.0, "discharge_mw": 0.0},
            constraints=_constraints(),
            error_bound_mwh=5.0,
        )
        assert res["observed_safe"] is True
        assert res["true_safe_if_bound_holds"] is True

    def test_matches_basic_at_zero_error(self):
        action = {"charge_mw": 5.0, "discharge_mw": 0.0}
        basic = check_soc_invariance(40.0, action, _constraints())
        tight = check_tightened_soc_invariance(
            current_soc_obs=40.0, action=action, constraints=_constraints(), error_bound_mwh=0.0,
        )
        assert tight["observed_safe"] is basic
        assert tight["true_safe_if_bound_holds"] is basic

    def test_reliability_based_error(self):
        res = check_tightened_soc_invariance(
            current_soc_obs=50.0,
            action={"charge_mw": 5.0, "discharge_mw": 0.0},
            constraints=_constraints(),
            reliability_w=0.5,
            max_error_mwh=10.0,
        )
        assert "error_bound_mwh" in res
        assert res["error_bound_mwh"] > 0.0


class TestSafetyFilterSummary:
    def test_includes_projection_flag(self):
        res = safety_filter_projection_summary(
            current_soc_obs=50.0,
            action={"charge_mw": 5.0, "discharge_mw": 0.0},
            constraints=_constraints(),
            reliability_w=0.8,
            max_error_mwh=5.0,
        )
        assert res["projection_interpretable_as_safety_filter"] is True
        assert "observed_safe" in res
