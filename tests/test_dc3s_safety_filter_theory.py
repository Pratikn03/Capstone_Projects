from __future__ import annotations

from gridpulse.dc3s.guarantee_checks import check_soc_invariance
from gridpulse.dc3s.safety_filter_theory import (
    check_tightened_soc_invariance,
    reliability_error_bound,
    tightened_soc_bounds,
)


def _constraints() -> dict[str, float]:
    return {
        "min_soc_mwh": 10.0,
        "max_soc_mwh": 90.0,
        "capacity_mwh": 100.0,
        "time_step_hours": 1.0,
        "charge_efficiency": 0.95,
        "discharge_efficiency": 0.95,
    }


def test_reliability_error_bound_is_monotone() -> None:
    clean = reliability_error_bound(reliability_w=0.95, max_error_mwh=10.0, min_error_mwh=1.0)
    degraded = reliability_error_bound(reliability_w=0.25, max_error_mwh=10.0, min_error_mwh=1.0)
    assert degraded > clean


def test_tightened_bounds_shrink_with_larger_error() -> None:
    narrow = tightened_soc_bounds(min_soc_mwh=10.0, max_soc_mwh=90.0, error_bound_mwh=2.0)
    wide = tightened_soc_bounds(min_soc_mwh=10.0, max_soc_mwh=90.0, error_bound_mwh=8.0)
    assert wide[0] > narrow[0]
    assert wide[1] < narrow[1]


def test_tightened_observed_feasibility_implies_true_feasibility_under_bound() -> None:
    result = check_tightened_soc_invariance(
        current_soc_obs=50.0,
        action={"charge_mw": 5.0, "discharge_mw": 0.0},
        constraints=_constraints(),
        error_bound_mwh=5.0,
    )
    assert result["observed_safe"] is True
    assert result["true_safe_if_bound_holds"] is True


def test_tightened_check_matches_basic_invariance_when_error_is_zero() -> None:
    action = {"charge_mw": 5.0, "discharge_mw": 0.0}
    basic = check_soc_invariance(40.0, action, _constraints())
    tightened = check_tightened_soc_invariance(
        current_soc_obs=40.0,
        action=action,
        constraints=_constraints(),
        error_bound_mwh=0.0,
    )
    assert tightened["observed_safe"] is basic
    assert tightened["true_safe_if_bound_holds"] is basic
