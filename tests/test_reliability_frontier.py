"""
Tests for the reference T10 reliability frontier utilities.

Verifies:
  1. Frontier computation (upper/lower bound monotonicity and T3 formula)
  2. Minimum reliability computation (closed-form inversion of T3)
  3. DC3S optimality check
  4. Frontier summary formatting
  5. Edge cases (T=1, alpha near 0/1, target_violations=0)
"""

import math

import numpy as np
import pytest

from orius.universal.reliability_safety_frontier import (
    achieved_violation_rate,
    compute_frontier,
    frontier_summary,
    is_dc3s_optimal,
    minimum_reliability_for_target,
)


class TestComputeFrontier:
    """Tests for compute_frontier()."""

    def test_returns_correct_number_of_points(self):
        points = compute_frontier(alpha=0.05, T=2000)
        assert len(points) == 101  # default 101-point grid

    def test_upper_bound_formula_is_correct(self):
        """Upper bound must be α(1−w̄)T for each point."""
        alpha, T = 0.05, 2000
        points = compute_frontier(alpha=alpha, T=T)
        for p in points:
            expected = alpha * (1.0 - p.mean_reliability) * T
            assert abs(p.upper_bound - expected) < 1e-9, (
                f"Upper bound mismatch at w̄={p.mean_reliability}: got {p.upper_bound}, expected {expected}"
            )

    def test_upper_bound_monotone_decreasing_in_reliability(self):
        """Higher reliability → fewer expected violations (T3 is monotone in w̄)."""
        points = compute_frontier(alpha=0.05, T=2000)
        uppers = [p.upper_bound for p in points]
        for i in range(len(uppers) - 1):
            assert uppers[i] >= uppers[i + 1] - 1e-9, (
                f"Upper bound not monotone at index {i}: {uppers[i]} < {uppers[i + 1]}"
            )

    def test_lower_bound_le_upper_bound(self):
        """Lower bound ≤ upper bound at every point (T10 is a lower bound on T3)."""
        points = compute_frontier(alpha=0.05, T=2000)
        for p in points:
            assert p.lower_bound <= p.upper_bound + 1e-9

    def test_lower_bound_nonnegative(self):
        """Violation counts cannot be negative."""
        points = compute_frontier(alpha=0.05, T=2000)
        for p in points:
            assert p.lower_bound >= 0.0

    def test_at_w_bar_one_both_bounds_are_zero(self):
        """At perfect reliability (w̄=1), both bounds equal 0."""
        points = compute_frontier(alpha=0.05, T=2000, w_bar_values=np.array([1.0]))
        assert len(points) == 1
        p = points[0]
        assert abs(p.upper_bound) < 1e-9
        assert abs(p.lower_bound) < 1e-9

    def test_at_w_bar_zero_upper_bound_is_alpha_T(self):
        """At zero reliability (w̄=0), upper bound = α·T."""
        alpha, T = 0.05, 2000
        points = compute_frontier(alpha=alpha, T=T, w_bar_values=np.array([0.0]))
        assert abs(points[0].upper_bound - alpha * T) < 1e-9

    def test_gap_equals_correction_term(self):
        """Gap = upper − lower = correction = 2·√(T·log T)."""
        alpha, T = 0.05, 2000
        expected_correction = 2.0 * math.sqrt(T * math.log(T))
        points = compute_frontier(alpha=alpha, T=T)
        for p in points:
            if p.upper_bound >= expected_correction:
                # In the regime where lower > 0, gap should equal correction.
                assert abs(p.gap - expected_correction) < 1e-6

    def test_custom_w_bar_values(self):
        """Custom w̄ grid is used correctly."""
        w_vals = np.array([0.0, 0.5, 1.0])
        points = compute_frontier(alpha=0.05, T=1000, w_bar_values=w_vals)
        assert len(points) == 3
        assert points[0].mean_reliability == 0.0
        assert points[1].mean_reliability == 0.5
        assert points[2].mean_reliability == 1.0

    def test_raises_on_invalid_alpha(self):
        with pytest.raises(ValueError, match="alpha"):
            compute_frontier(alpha=0.0, T=100)
        with pytest.raises(ValueError, match="alpha"):
            compute_frontier(alpha=1.5, T=100)

    def test_raises_on_invalid_T(self):
        with pytest.raises(ValueError, match="T"):
            compute_frontier(alpha=0.05, T=0)


class TestMinimumReliabilityForTarget:
    """Tests for minimum_reliability_for_target()."""

    def test_zero_target_requires_perfect_reliability(self):
        w_star = minimum_reliability_for_target(alpha=0.05, T=2000, target_violations=0.0)
        assert w_star == 1.0

    def test_formula_is_correct_inversion_of_T3(self):
        """w̄* = 1 − target / (α · T)."""
        alpha, T = 0.05, 2000
        for target in [1.0, 5.0, 10.0, 50.0]:
            w_star = minimum_reliability_for_target(alpha=alpha, T=T, target_violations=target)
            expected = 1.0 - target / (alpha * T)
            assert abs(w_star - expected) < 1e-9

    def test_large_target_returns_zero(self):
        """If target ≥ α·T, any reliability (even 0) suffices."""
        w_star = minimum_reliability_for_target(alpha=0.05, T=100, target_violations=100.0)
        assert w_star == 0.0

    def test_negative_target_treated_as_zero(self):
        """Negative target is treated as zero → returns 1.0."""
        w_star = minimum_reliability_for_target(alpha=0.05, T=2000, target_violations=-1.0)
        assert w_star == 1.0

    def test_round_trip_with_frontier(self):
        """minimum_reliability_for_target should be inverse of upper_bound at that w̄."""
        alpha, T = 0.05, 2000
        target = 5.0
        w_star = minimum_reliability_for_target(alpha=alpha, T=T, target_violations=target)
        # Reconstruct upper bound at w_star
        reconstructed_upper = alpha * (1.0 - w_star) * T
        assert abs(reconstructed_upper - target) < 1e-6


class TestIsDC3SOptimal:
    """Tests for is_dc3s_optimal()."""

    def test_zero_violations_is_optimal(self):
        """Zero violations is always within the bound."""
        assert is_dc3s_optimal(alpha=0.05, T=2000, observed_violations=0.0, mean_reliability=0.9)

    def test_violations_at_upper_bound_is_optimal(self):
        """Exactly at the upper bound (with default 2% tolerance) is optimal."""
        alpha, T, w_bar = 0.05, 2000, 0.9
        upper = alpha * (1.0 - w_bar) * T  # = 10.0
        assert is_dc3s_optimal(alpha=alpha, T=T, observed_violations=upper, mean_reliability=w_bar)

    def test_violations_far_above_bound_is_not_optimal(self):
        """Violations >> upper bound indicates a bug in the shield."""
        alpha, T, w_bar = 0.05, 2000, 0.9
        upper = alpha * (1.0 - w_bar) * T  # = 10.0
        assert not is_dc3s_optimal(alpha=alpha, T=T, observed_violations=upper * 2.0, mean_reliability=w_bar)

    def test_tolerance_parameter(self):
        alpha, T, w_bar = 0.05, 2000, 0.9
        upper = alpha * (1.0 - w_bar) * T
        # 5% above bound: passes with 10% tolerance, fails with 2% tolerance.
        slightly_above = upper * 1.05
        assert is_dc3s_optimal(alpha, T, slightly_above, w_bar, upper_tolerance=0.10)
        assert not is_dc3s_optimal(alpha, T, slightly_above, w_bar, upper_tolerance=0.02)


class TestAchievedViolationRate:
    def test_tsvr_formula(self):
        assert abs(achieved_violation_rate(10.0, 2000) - 0.005) < 1e-9

    def test_zero_violations_zero_rate(self):
        assert achieved_violation_rate(0.0, 2000) == 0.0

    def test_safe_division_by_zero(self):
        """T=0 should not raise, returns 0."""
        assert achieved_violation_rate(0.0, 0) == 0.0


class TestFrontierSummary:
    def test_summary_contains_key_values(self):
        summary = frontier_summary(alpha=0.05, T=2000)
        assert "Reliability-Safety Frontier" in summary
        assert "α=0.05" in summary
        assert "T=2000" in summary
        assert "w̄*" in summary

    def test_summary_is_string(self):
        assert isinstance(frontier_summary(0.05, 100), str)
