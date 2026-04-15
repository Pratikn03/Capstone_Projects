"""Comprehensive tests for DC3S coverage theorem verification."""
from __future__ import annotations

import numpy as np
import pytest

from orius.dc3s.coverage_theorem import (
    assert_coverage_guarantee,
    compute_empirical_coverage,
    compute_expected_violation_bound,
    evaluate_empirical_core_bound,
    inflation_lower_bound,
    verify_inflation_geq_one,
)


class TestVerifyInflation:
    def test_passes_above_one(self):
        verify_inflation_geq_one(1.5)

    def test_passes_at_exactly_one(self):
        verify_inflation_geq_one(1.0)

    def test_raises_below_one(self):
        with pytest.raises(ValueError, match=">= 1"):
            verify_inflation_geq_one(0.5)

    def test_boundary_tolerance(self):
        verify_inflation_geq_one(1.0 - 1e-10)

    def test_raises_well_below(self):
        with pytest.raises(ValueError):
            verify_inflation_geq_one(0.0)


class TestEmpiricalCoverage:
    def test_perfect_coverage(self):
        y = np.array([10.0, 20.0, 30.0])
        lo = np.array([5.0, 15.0, 25.0])
        hi = np.array([15.0, 25.0, 35.0])
        res = compute_empirical_coverage(y, lo, hi)
        assert res["picp"] == pytest.approx(1.0)
        assert res["n_samples"] == 3

    def test_zero_coverage(self):
        y = np.array([1.0, 2.0, 3.0])
        lo = np.array([10.0, 20.0, 30.0])
        hi = np.array([15.0, 25.0, 35.0])
        res = compute_empirical_coverage(y, lo, hi)
        assert res["picp"] == pytest.approx(0.0)

    def test_partial_coverage(self):
        y = np.array([10.0, 100.0])
        lo = np.array([5.0, 5.0])
        hi = np.array([15.0, 15.0])
        res = compute_empirical_coverage(y, lo, hi)
        assert res["picp"] == pytest.approx(0.5)

    def test_mean_width_correct(self):
        lo = np.array([0.0, 10.0])
        hi = np.array([20.0, 30.0])
        res = compute_empirical_coverage(np.array([10.0, 20.0]), lo, hi)
        assert res["mean_width"] == pytest.approx(20.0)

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="same number"):
            compute_empirical_coverage(np.array([1.0]), np.array([0.0, 1.0]), np.array([2.0]))

    def test_lower_gt_upper_raises(self):
        with pytest.raises(ValueError, match="lower must be"):
            compute_empirical_coverage(np.array([1.0]), np.array([10.0]), np.array([5.0]))

    def test_2d_arrays_flattened(self):
        y = np.array([[10.0, 20.0]])
        lo = np.array([[5.0, 15.0]])
        hi = np.array([[15.0, 25.0]])
        res = compute_empirical_coverage(y, lo, hi)
        assert res["n_samples"] == 2


class TestAssertCoverageGuarantee:
    def test_passes_when_picp_sufficient(self):
        y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        lo = np.array([5.0, 15.0, 25.0, 35.0, 45.0])
        hi = np.array([15.0, 25.0, 35.0, 45.0, 55.0])
        res = assert_coverage_guarantee(y, lo, hi, alpha=0.10, tolerance=0.02)
        assert res["passed"] is True

    def test_raises_when_picp_too_low(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        lo = np.array([10.0, 10.0, 10.0, 10.0])
        hi = np.array([20.0, 20.0, 20.0, 20.0])
        with pytest.raises(AssertionError, match="Empirical coverage="):
            assert_coverage_guarantee(y, lo, hi, alpha=0.10, tolerance=0.01)

    def test_tolerance_allows_slack(self):
        y = np.array([10.0, 100.0, 20.0, 30.0])
        lo = np.array([5.0, 5.0, 15.0, 25.0])
        hi = np.array([15.0, 15.0, 25.0, 35.0])
        res = assert_coverage_guarantee(y, lo, hi, alpha=0.10, tolerance=0.20)
        assert res["passed"] is True


class TestInflationLowerBound:
    def test_always_one_for_valid_params(self):
        assert inflation_lower_bound(k_quality=1.5, k_drift=2.0) == pytest.approx(1.0)

    def test_positive_params(self):
        assert inflation_lower_bound(k_quality=0.0, k_drift=0.0) == 1.0

    def test_with_w_t_min(self):
        val = inflation_lower_bound(k_quality=1.0, k_drift=1.0, w_t_min=0.1)
        assert val >= 1.0


class TestExpectedViolationBound:
    def test_formula(self):
        res = compute_expected_violation_bound([1.0, 0.8, 0.6], alpha=0.10)
        w_bar = (1.0 + 0.8 + 0.6) / 3.0
        assert res["bound_expected_violations"] == pytest.approx(0.10 * (1.0 - w_bar) * 3)
        assert res["horizon"] == 3

    def test_perfect_reliability_zero_bound(self):
        res = compute_expected_violation_bound([1.0, 1.0, 1.0], alpha=0.10)
        assert res["bound_expected_violations"] == pytest.approx(0.0)

    def test_zero_reliability(self):
        res = compute_expected_violation_bound([0.0, 0.0], alpha=0.10)
        assert res["bound_expected_violations"] == pytest.approx(0.10 * 2.0)

    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError, match="alpha"):
            compute_expected_violation_bound([0.5], alpha=1.5)

    def test_invalid_reliability_raises(self):
        with pytest.raises(ValueError, match="reliability"):
            compute_expected_violation_bound([1.5], alpha=0.1)

    def test_bound_tsvr_correct(self):
        res = compute_expected_violation_bound([0.5, 0.5, 0.5, 0.5], alpha=0.10)
        assert res["bound_tsvr"] == pytest.approx(res["bound_expected_violations"] / 4.0)


class TestEvaluateEmpiricalCoreBound:
    def test_zero_violations_passes(self):
        res = evaluate_empirical_core_bound([0, 0, 0], [1.0, 0.8, 0.7], alpha=0.10)
        assert res["passed"] is True
        assert res["empirical_violation_count"] == 0.0

    def test_violations_exceed_bound_fails(self):
        res = evaluate_empirical_core_bound([1, 1, 1], [0.9, 0.9, 0.9], alpha=0.10)
        assert res["passed"] is False

    def test_slack_violations_relaxes_bound(self):
        res = evaluate_empirical_core_bound([1, 0, 0], [0.9, 0.9, 0.9], alpha=0.10, slack_violations=2.0)
        assert res["passed"] is True

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="same length"):
            evaluate_empirical_core_bound([0, 0], [1.0], alpha=0.10)

    def test_empirical_tsvr(self):
        res = evaluate_empirical_core_bound([1, 0, 0, 0], [0.5, 0.5, 0.5, 0.5], alpha=0.10)
        assert res["empirical_tsvr"] == pytest.approx(0.25)

    def test_negative_slack_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            evaluate_empirical_core_bound([0], [1.0], alpha=0.10, slack_violations=-1.0)
