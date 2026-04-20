"""Tests for legacy auxiliary coverage helpers.

These helpers remain useful for subgroup-coverage and concentration diagnostics,
but they are not the active T9/T10 theorem surface.

Phase 2 of ORIUS gap-closing plan:
  - mondrian_group_coverage: per-reliability-bin PICP verification
  - hoeffding_violation_bound: high-probability tail bound on TSVR
  - evaluate_group_conditional_coverage: record-list API
"""
from __future__ import annotations

import math
import numpy as np
import pytest

from orius.dc3s.coverage_theorem import (
    mondrian_group_coverage,
    hoeffding_violation_bound,
    evaluate_group_conditional_coverage,
    compute_expected_violation_bound,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_perfect_intervals(n: int, rng_seed: int = 0) -> tuple:
    """Return intervals that always cover y_true (picp=1.0)."""
    rng = np.random.default_rng(rng_seed)
    y_true = rng.normal(0, 1, n)
    lower = y_true - 2.0
    upper = y_true + 2.0
    w = rng.uniform(0.2, 1.0, n)
    return y_true, lower, upper, w


def _make_partial_intervals(n: int, coverage_frac: float = 0.90, rng_seed: int = 42) -> tuple:
    """Return fixed-centre intervals covering ~coverage_frac of N(0,1) y_true values."""
    rng = np.random.default_rng(rng_seed)
    y_true = rng.normal(0, 1, n)
    # Fixed interval [-1.645, 1.645] covers ~90% of N(0,1) regardless of y_true
    lower = np.full(n, -1.645)
    upper = np.full(n,  1.645)
    w = rng.uniform(0.0, 1.0, n)
    return y_true, lower, upper, w


# ---------------------------------------------------------------------------
# mondrian_group_coverage
# ---------------------------------------------------------------------------

class TestMondrianGroupCoverage:
    def test_perfect_coverage_all_bins_pass(self):
        y, lo, hi, w = _make_perfect_intervals(300)
        result = mondrian_group_coverage(y, lo, hi, w, n_bins=3, alpha=0.10)
        assert result["all_pass"] is True
        assert result["overall_picp"] == pytest.approx(1.0)
        assert len(result["groups"]) == 3

    def test_group_count_matches_n_bins(self):
        y, lo, hi, w = _make_perfect_intervals(100)
        for k in [2, 3, 4]:
            result = mondrian_group_coverage(y, lo, hi, w, n_bins=k)
            assert len(result["groups"]) == k

    def test_each_group_has_w_range(self):
        y, lo, hi, w = _make_perfect_intervals(200)
        result = mondrian_group_coverage(y, lo, hi, w, n_bins=3)
        for g in result["groups"]:
            assert g["w_lo"] <= g["w_hi"]

    def test_partial_coverage_90pct_alpha10(self):
        """~90% coverage intervals should pass at alpha=0.10 (target=0.90, tol=0.02)."""
        y, lo, hi, w = _make_partial_intervals(5000, coverage_frac=0.90)
        result = mondrian_group_coverage(y, lo, hi, w, n_bins=3, alpha=0.10)
        # Overall picp should be near 90%
        assert result["overall_picp"] == pytest.approx(0.90, abs=0.03)
        assert result["all_pass"] is True  # each bin should also be ~0.90

    def test_very_low_coverage_fails(self):
        """Fixed interval [-0.1, 0.1] covers ~8% of N(0,1) → all groups fail at alpha=0.10."""
        rng = np.random.default_rng(0)
        n = 1000
        y = rng.normal(0, 1, n)
        lo = np.full(n, -0.1)   # fixed tiny interval, NOT centred on y
        hi = np.full(n,  0.1)   # ~8% coverage for N(0,1)
        w = rng.uniform(0, 1, n)
        result = mondrian_group_coverage(y, lo, hi, w, n_bins=3, alpha=0.10)
        assert result["all_pass"] is False

    def test_n_samples_sum_to_N(self):
        y, lo, hi, w = _make_perfect_intervals(150)
        result = mondrian_group_coverage(y, lo, hi, w, n_bins=3)
        total_n = sum(g["n"] for g in result["groups"])
        assert total_n == 150

    def test_raises_mismatched_lengths(self):
        y = np.ones(10)
        lo = np.ones(9)
        hi = np.ones(10)
        w  = np.ones(10)
        with pytest.raises(ValueError, match="same length"):
            mondrian_group_coverage(y, lo, hi, w)

    def test_returns_alpha_and_n_bins(self):
        y, lo, hi, w = _make_perfect_intervals(60)
        result = mondrian_group_coverage(y, lo, hi, w, n_bins=4, alpha=0.05)
        assert result["alpha"] == pytest.approx(0.05)
        assert result["n_bins"] == 4

    def test_empty_bins_fail_instead_of_passing_vacuously(self):
        y = np.array([0.0, 0.0, 0.0])
        lo = np.array([-1.0, -1.0, -1.0])
        hi = np.array([1.0, 1.0, 1.0])
        w = np.array([0.5, 0.5, 0.5])
        result = mondrian_group_coverage(y, lo, hi, w, n_bins=3, alpha=0.10)
        assert result["empty_bins"] == 2
        assert result["all_pass"] is False

    def test_accepts_lists(self):
        y, lo, hi, w = _make_perfect_intervals(50)
        result = mondrian_group_coverage(y.tolist(), lo.tolist(), hi.tolist(), w.tolist())
        assert result["overall_picp"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# hoeffding_violation_bound
# ---------------------------------------------------------------------------

class TestHoeffdingViolationBound:
    def test_basic_values(self):
        result = hoeffding_violation_bound(T=100, alpha=0.10, w_bar=0.80, epsilon=0.05)
        # expectation_bound = 0.10 * (1 - 0.80) = 0.02
        assert result["expectation_bound"] == pytest.approx(0.02)
        # high_prob_bound = 0.02 + 0.05 = 0.07
        assert result["high_prob_bound"] == pytest.approx(0.07)
        # tail = exp(-2 * 100 * 0.0025) = exp(-0.5) ≈ 0.6065
        assert result["tail_probability"] == pytest.approx(math.exp(-0.5), rel=1e-6)

    def test_large_T_gives_small_tail(self):
        result = hoeffding_violation_bound(T=10000, alpha=0.10, w_bar=0.70, epsilon=0.10)
        # exp(-2 * 10000 * 0.01) = exp(-200) ≈ 0
        assert result["tail_probability"] < 1e-80

    def test_small_epsilon_gives_large_tail(self):
        result = hoeffding_violation_bound(T=10, alpha=0.10, w_bar=0.50, epsilon=0.001)
        # exp(-2 * 10 * 0.000001) ≈ 1
        assert result["tail_probability"] > 0.99

    def test_consistency_with_expected_bound(self):
        """high_prob_bound = expectation_bound + epsilon."""
        result = hoeffding_violation_bound(T=200, alpha=0.05, w_bar=0.60, epsilon=0.03)
        assert result["high_prob_bound"] == pytest.approx(
            result["expectation_bound"] + result["epsilon"], rel=1e-9
        )

    def test_invalid_T_raises(self):
        with pytest.raises(ValueError, match="positive"):
            hoeffding_violation_bound(T=0, alpha=0.10, w_bar=0.5, epsilon=0.1)

    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError, match="alpha"):
            hoeffding_violation_bound(T=100, alpha=1.5, w_bar=0.5, epsilon=0.1)

    def test_invalid_w_bar_raises(self):
        with pytest.raises(ValueError, match="w_bar"):
            hoeffding_violation_bound(T=100, alpha=0.1, w_bar=-0.1, epsilon=0.05)

    def test_invalid_epsilon_raises(self):
        with pytest.raises(ValueError, match="epsilon"):
            hoeffding_violation_bound(T=100, alpha=0.1, w_bar=0.5, epsilon=0.0)

    def test_tail_in_0_1(self):
        result = hoeffding_violation_bound(T=50, alpha=0.10, w_bar=0.75, epsilon=0.02)
        assert 0.0 <= result["tail_probability"] <= 1.0

    def test_returns_all_keys(self):
        result = hoeffding_violation_bound(T=48, alpha=0.10, w_bar=0.80, epsilon=0.05)
        for key in ("expectation_bound", "high_prob_bound", "tail_probability",
                    "T", "alpha", "w_bar", "epsilon"):
            assert key in result


# ---------------------------------------------------------------------------
# evaluate_group_conditional_coverage (record-list API)
# ---------------------------------------------------------------------------

class TestEvaluateGroupConditionalCoverage:
    def _make_records(self, n: int, perfect: bool = True, rng_seed: int = 0) -> list:
        rng = np.random.default_rng(rng_seed)
        y = rng.normal(0, 1, n)
        w = rng.uniform(0.2, 1.0, n)
        if perfect:
            lo, hi = y - 2.0, y + 2.0
        else:
            lo, hi = y - 0.05, y + 0.05  # negligible coverage
        return [
            {"y_true": float(y[i]), "lower": float(lo[i]),
             "upper": float(hi[i]), "reliability_w": float(w[i])}
            for i in range(n)
        ]

    def test_perfect_records_pass(self):
        records = self._make_records(200, perfect=True)
        result = evaluate_group_conditional_coverage(records, alpha=0.10, n_bins=3)
        assert result["all_pass"] is True
        assert result["overall_picp"] == pytest.approx(1.0)

    def test_empty_records_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            evaluate_group_conditional_coverage([])

    def test_missing_key_raises(self):
        records = [{"y_true": 1.0, "lower": 0.0, "upper": 2.0}]  # missing reliability_w
        with pytest.raises(KeyError):
            evaluate_group_conditional_coverage(records)

    def test_groups_structure(self):
        records = self._make_records(120)
        result = evaluate_group_conditional_coverage(records, n_bins=3)
        assert len(result["groups"]) == 3
        for g in result["groups"]:
            assert "picp" in g
            assert "passed" in g
            assert "n" in g


# ---------------------------------------------------------------------------
# Integration: empirical TSVR <= Hoeffding high_prob_bound for real traces
# ---------------------------------------------------------------------------

class TestHoeffdingIntegration:
    def test_orius_tsvr_within_hoeffding_bound(self):
        """Simulate a DC3S run and verify TSVR <= Hoeffding bound with epsilon=0.10."""
        rng = np.random.default_rng(99)
        T = 240
        # Simulate reliability scores and violations
        w = rng.uniform(0.7, 1.0, T)  # high reliability
        # Under DC3S, violations occur rarely (proportional to 1-w)
        violations = rng.uniform(0, 1, T) < (0.10 * (1 - w))
        tsvr = float(np.mean(violations))
        w_bar = float(np.mean(w))

        alpha = 0.10
        epsilon = 0.10
        bound = hoeffding_violation_bound(T=T, alpha=alpha, w_bar=w_bar, epsilon=epsilon)
        # The empirical TSVR should be near or below the expectation bound
        assert tsvr <= bound["high_prob_bound"] + 0.05  # generous check for stochastic test

    def test_expectation_bound_consistency(self):
        """Hoeffding expectation_bound matches compute_expected_violation_bound bound_tsvr."""
        rng = np.random.default_rng(7)
        w = rng.uniform(0.5, 1.0, 100)
        w_bar = float(np.mean(w))
        alpha = 0.10

        evb = compute_expected_violation_bound(w, alpha=alpha)
        hb  = hoeffding_violation_bound(T=100, alpha=alpha, w_bar=w_bar, epsilon=0.05)

        assert evb["bound_tsvr"] == pytest.approx(hb["expectation_bound"], abs=1e-6)
