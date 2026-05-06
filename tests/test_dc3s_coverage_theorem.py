import numpy as np
import pytest

from orius.dc3s.coverage_theorem import (
    assert_coverage_guarantee,
    compute_empirical_coverage,
    compute_expected_violation_bound,
    evaluate_empirical_core_bound,
    get_core_bound_assumptions,
    inflation_lower_bound,
    verify_inflation_geq_one,
)


def test_verify_inflation_geq_one():
    verify_inflation_geq_one(1.5)
    verify_inflation_geq_one(1.0)
    with pytest.raises(ValueError, match=">= 1"):
        verify_inflation_geq_one(0.9)


def test_compute_empirical_coverage():
    y_true = np.array([10.0, 15.0, 20.0, 25.0])
    lower = np.array([8.0, 16.0, 18.0, 26.0])
    upper = np.array([12.0, 18.0, 22.0, 28.0])
    # 10 is in [8, 12] -> True
    # 15 is in [16, 18] -> False (15 < 16)
    # 20 is in [18, 22] -> True
    # 25 is in [26, 28] -> False (25 < 26)
    # Coverage = 2/4 = 0.5
    res = compute_empirical_coverage(y_true, lower, upper)
    assert np.isclose(res["picp"], 0.5)
    assert res["n_samples"] == 4


def test_assert_coverage_guarantee():
    y_true = np.array([10.0, 15.0, 20.0, 25.0])
    lower = np.array([8.0, 14.0, 18.0, 26.0])
    upper = np.array([12.0, 18.0, 22.0, 28.0])
    # 10 in [8, 12] -> T
    # 15 in [14, 18] -> T
    # 20 in [18, 22] -> T
    # 25 in [26, 28] -> F (25 < 26)
    # Coverage = 3/4 = 0.75
    # Target alpha=0.20 -> 0.80. Tolerance=0.05. Minimum needed = 0.75. Should pass.
    res = assert_coverage_guarantee(y_true, lower, upper, alpha=0.20, tolerance=0.05)
    assert res["passed"] is True

    with pytest.raises(AssertionError):
        # Target alpha=0.10 -> 0.90. Tolerance=0.01. Minimum needed = 0.89. Fails with 0.75.
        assert_coverage_guarantee(y_true, lower, upper, alpha=0.10, tolerance=0.01)


def test_inflation_lower_bound():
    res = inflation_lower_bound(k_quality=1.5, k_drift=2.0)
    assert np.isclose(res, 1.0)


def test_compute_expected_violation_bound():
    summary = compute_expected_violation_bound([1.0, 0.8, 0.7], alpha=0.10)
    assert summary["horizon"] == 3
    assert np.isclose(summary["mean_reliability_w"], (1.0 + 0.8 + 0.7) / 3.0)
    assert np.isclose(summary["bound_expected_violations"], 0.05)
    assert np.isclose(summary["bound_tsvr"], 0.05 / 3.0)
    assert "assumptions_used" in summary
    assert "risk-budget contract" in summary["interpretation"]


def test_core_bound_assumptions_explicitly_rule_out_probability_reading():
    assumptions = get_core_bound_assumptions()
    assert "w_t is a runtime reliability score, not a probability by definition." in assumptions
    assert any("predictable per-step residual-risk budget" in item for item in assumptions)


def test_expected_violation_bound_surfaces_same_assumption_list():
    assumptions = get_core_bound_assumptions()
    summary = compute_expected_violation_bound([0.9, 0.7], alpha=0.10)
    assert tuple(summary["assumptions_used"]) == assumptions


def test_evaluate_empirical_core_bound():
    passing = evaluate_empirical_core_bound([0, 0, 0], [1.0, 0.8, 0.7], alpha=0.10)
    assert passing["passed"] is True
    assert np.isclose(passing["empirical_violation_count"], 0.0)

    failing = evaluate_empirical_core_bound([1, 0, 0], [1.0, 0.8, 0.7], alpha=0.10)
    assert failing["passed"] is False
    assert np.isclose(failing["empirical_violation_count"], 1.0)
