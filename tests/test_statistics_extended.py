"""Regression tests for the extended statistics helpers."""

from __future__ import annotations

import numpy as np
import pytest

from orius.evaluation.stats import (
    bca_bootstrap,
    benjamini_hochberg,
    bonferroni,
    mcnemar_test,
    paired_bootstrap,
    wilcoxon_signed_rank,
)


def test_bca_bootstrap_contains_point_estimate_for_simple_sample() -> None:
    sample = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = bca_bootstrap(sample, n_bootstrap=500, seed=7)
    assert result["ci_lower"] <= result["point_estimate"] <= result["ci_upper"]
    assert result["confidence"] == pytest.approx(0.95)


def test_paired_bootstrap_estimates_positive_effect() -> None:
    baseline = np.array([10.0, 10.5, 11.0, 9.5, 10.2])
    treatment = baseline + np.array([0.5, 0.4, 0.6, 0.7, 0.3])
    result = paired_bootstrap(baseline, treatment, n_bootstrap=500, seed=11)
    assert result["effect_size"] > 0.0
    assert result["ci_lower"] <= result["effect_size"] <= result["ci_upper"]


def test_wilcoxon_signed_rank_reports_no_difference_for_equal_samples() -> None:
    sample = np.array([1.0, 2.0, 3.0, 4.0])
    result = wilcoxon_signed_rank(sample, sample.copy())
    assert result["p_value"] == pytest.approx(1.0)
    assert result["significant"] is False


def test_mcnemar_test_uses_exact_mode_for_small_tables() -> None:
    table = np.array([[10, 1], [5, 8]])
    result = mcnemar_test(table)
    assert result["method"] == "exact"
    assert 0.0 <= result["p_value"] <= 1.0


def test_benjamini_hochberg_and_bonferroni_return_expected_rejections() -> None:
    p_values = [0.001, 0.01, 0.03, 0.20]
    bh = benjamini_hochberg(p_values, alpha=0.05)
    bon = bonferroni(p_values, alpha=0.05)
    assert bh["n_rejected"] >= bon["n_rejected"]
    assert bh["rejected"][0] is True
    assert bon["rejected"][0] is True
