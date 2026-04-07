"""Property-based tests for the executable theorem witnesses."""
from __future__ import annotations

import math

from hypothesis import given, settings
from hypothesis import strategies as st

from orius.dc3s.theoretical_guarantees import (
    compute_adaptive_regret_bound,
    compute_finite_sample_coverage_bound,
    compute_separation_gap,
)


@settings(deadline=None, max_examples=80)
@given(
    n_calibration=st.integers(min_value=1, max_value=20_000),
    alpha=st.floats(min_value=0.01, max_value=0.40, allow_nan=False, allow_infinity=False),
    delta=st.floats(min_value=0.01, max_value=0.40, allow_nan=False, allow_infinity=False),
    w_min=st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False),
)
def test_t9_effective_sample_size_matches_floor_rule(
    n_calibration: int,
    alpha: float,
    delta: float,
    w_min: float,
) -> None:
    result = compute_finite_sample_coverage_bound(
        n_calibration=n_calibration,
        alpha=alpha,
        delta=delta,
        w_min=w_min,
    )
    assert result["n_eff"] == max(1, math.floor(n_calibration * w_min))
    assert result["coverage_bound"] <= result["nominal_coverage"]
    assert result["epsilon"] >= 0.0


@settings(deadline=None, max_examples=60)
@given(
    n_small=st.integers(min_value=5, max_value=5_000),
    n_extra=st.integers(min_value=1, max_value=5_000),
    alpha=st.floats(min_value=0.01, max_value=0.30, allow_nan=False, allow_infinity=False),
    delta=st.floats(min_value=0.01, max_value=0.30, allow_nan=False, allow_infinity=False),
    w_min=st.floats(min_value=0.05, max_value=1.0, allow_nan=False, allow_infinity=False),
)
def test_t9_more_calibration_data_never_loosens_bound(
    n_small: int,
    n_extra: int,
    alpha: float,
    delta: float,
    w_min: float,
) -> None:
    short = compute_finite_sample_coverage_bound(n_small, alpha, delta, w_min)
    long = compute_finite_sample_coverage_bound(n_small + n_extra, alpha, delta, w_min)
    assert long["epsilon"] <= short["epsilon"]
    assert long["coverage_bound"] >= short["coverage_bound"]


@settings(deadline=None, max_examples=80)
@given(
    dc3s_violations=st.floats(min_value=0.0, max_value=0.20, allow_nan=False, allow_infinity=False),
    dc3s_interventions=st.floats(min_value=0.0, max_value=0.40, allow_nan=False, allow_infinity=False),
    blind_violation_gap=st.floats(min_value=1.0e-6, max_value=0.30, allow_nan=False, allow_infinity=False),
    blind_intervention_gap=st.floats(min_value=0.0, max_value=0.30, allow_nan=False, allow_infinity=False),
    w_min=st.floats(min_value=0.05, max_value=1.0, allow_nan=False, allow_infinity=False),
    alpha=st.floats(min_value=0.01, max_value=0.30, allow_nan=False, allow_infinity=False),
)
def test_t10_pareto_flag_matches_constructed_dominance(
    dc3s_violations: float,
    dc3s_interventions: float,
    blind_violation_gap: float,
    blind_intervention_gap: float,
    w_min: float,
    alpha: float,
) -> None:
    blind_violations = dc3s_violations + blind_violation_gap
    blind_interventions = dc3s_interventions + blind_intervention_gap
    result = compute_separation_gap(
        dc3s_violations=dc3s_violations,
        dc3s_interventions=dc3s_interventions,
        blind_violations=blind_violations,
        blind_interventions=blind_interventions,
        w_min=w_min,
        alpha=alpha,
    )
    expected = blind_violation_gap > 0.0 or blind_intervention_gap > 0.0
    assert result.pareto_dominant is expected
    assert result.violation_lower_bound == alpha * (1.0 - w_min) / 2.0
    assert result.intervention_lower_bound == (1.0 - w_min) / 2.0


@settings(deadline=None, max_examples=60)
@given(
    t_small=st.integers(min_value=10, max_value=2_000),
    multiplier=st.integers(min_value=2, max_value=20),
    tau=st.floats(min_value=1.0, max_value=120.0, allow_nan=False, allow_infinity=False),
    max_oracle_jump=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)
def test_t11_per_step_regret_is_sublinear_in_horizon(
    t_small: int,
    multiplier: int,
    tau: float,
    max_oracle_jump: float,
) -> None:
    short = compute_adaptive_regret_bound(T=t_small, tau=tau, max_oracle_jump=max_oracle_jump)
    long = compute_adaptive_regret_bound(T=t_small * multiplier, tau=tau, max_oracle_jump=max_oracle_jump)
    assert long["per_step_bound"] <= short["per_step_bound"] + 1e-12
