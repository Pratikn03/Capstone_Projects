"""Property-based tests for the current ORIUS theorem witnesses."""
from __future__ import annotations

import pytest

from hypothesis import given, settings
from hypothesis import strategies as st

from orius.dc3s.theoretical_guarantees import (
    compute_stylized_frontier_lower_bound,
    compute_universal_impossibility_bound,
    evaluate_structural_transfer,
)


@settings(deadline=None, max_examples=80)
@given(
    horizon=st.integers(min_value=1, max_value=20_000),
    fault_rate=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    sensitivity_constant=st.floats(min_value=0.0, max_value=2.0, allow_nan=False, allow_infinity=False),
    usable_fraction=st.floats(min_value=0.05, max_value=1.0, allow_nan=False, allow_infinity=False),
)
def test_t9_impossibility_bound_matches_linear_scaling(
    horizon: int,
    fault_rate: float,
    sensitivity_constant: float,
    usable_fraction: float,
) -> None:
    result = compute_universal_impossibility_bound(
        horizon=horizon,
        fault_rate=fault_rate,
        sensitivity_constant=sensitivity_constant,
        usable_horizon_fraction=usable_fraction,
    )
    expected = horizon * usable_fraction * fault_rate * sensitivity_constant
    assert result["expected_lower_bound"] == pytest.approx(expected)
    assert result["linear_rate_lower_bound"] == pytest.approx(expected / horizon)


@settings(deadline=None, max_examples=80)
@given(
    reliability=st.lists(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=50,
    ),
    boundary_mass=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    alpha=st.floats(min_value=0.01, max_value=0.40, allow_nan=False, allow_infinity=False),
)
def test_t10_frontier_lower_bound_matches_half_sum_formula(
    reliability: list[float],
    boundary_mass: float,
    alpha: float,
) -> None:
    result = compute_stylized_frontier_lower_bound(
        reliability,
        boundary_mass=boundary_mass,
        alpha=alpha,
    )
    expected = 0.5 * boundary_mass * sum(1.0 - w for w in reliability)
    assert result["expected_lower_bound"] == pytest.approx(expected)
    if boundary_mass >= alpha / 2.0:
        assert result["special_case_active"] is True
    else:
        assert result["special_case_active"] is False


@settings(deadline=None, max_examples=60)
@given(
    coverage_holds=st.booleans(),
    sound_safe_action_set=st.booleans(),
    repair_membership_holds=st.booleans(),
    fallback_exists=st.booleans(),
    alpha=st.floats(min_value=0.01, max_value=0.40, allow_nan=False, allow_infinity=False),
)
def test_t11_transfer_requires_all_four_obligations(
    coverage_holds: bool,
    sound_safe_action_set: bool,
    repair_membership_holds: bool,
    fallback_exists: bool,
    alpha: float,
) -> None:
    result = evaluate_structural_transfer(
        coverage_holds=coverage_holds,
        sound_safe_action_set=sound_safe_action_set,
        repair_membership_holds=repair_membership_holds,
        fallback_exists=fallback_exists,
        alpha=alpha,
    )
    expected = coverage_holds and sound_safe_action_set and repair_membership_holds and fallback_exists
    assert result.one_step_transfer_holds is expected
    if expected:
        assert result.safety_probability_lower_bound == 1.0 - alpha
    else:
        assert result.counterexample is not None
