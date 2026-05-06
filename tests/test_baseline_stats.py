"""Functional tests for forecast comparison statistics."""

from __future__ import annotations

import numpy as np
import pytest

from orius.forecasting.stats import (
    cohens_d,
    diebold_mariano,
    holm_bonferroni,
    paired_block_bootstrap,
)


def _synthetic(seed: int = 0, n: int = 500, sigma_a: float = 1.0, sigma_b: float = 0.5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = rng.normal(0.0, 2.0, n)
    pred_a = y + rng.normal(0.0, sigma_a, n)
    pred_b = y + rng.normal(0.0, sigma_b, n)
    return y, pred_a, pred_b


class TestDieboldMariano:
    def test_detects_better_model_b(self) -> None:
        y, pa, pb = _synthetic(seed=1)
        result = diebold_mariano(y, pa, pb, horizon=24, loss="se")
        assert result.statistic > 0
        assert result.p_value < 0.05
        assert result.n > 0
        assert result.long_run_variance > 0

    def test_equal_models_not_significant(self) -> None:
        y, pa, _ = _synthetic(seed=2)
        rng = np.random.default_rng(99)
        pb = pa + rng.normal(0.0, 1e-8, pa.size)
        result = diebold_mariano(y, pa, pb, horizon=1, loss="se")
        assert result.p_value > 0.2

    def test_mae_loss_supported(self) -> None:
        y, pa, pb = _synthetic(seed=3)
        result = diebold_mariano(y, pa, pb, horizon=12, loss="ae")
        assert result.loss == "ae"
        assert result.statistic > 0

    def test_invalid_loss_raises(self) -> None:
        y, pa, pb = _synthetic(seed=4)
        with pytest.raises(ValueError):
            diebold_mariano(y, pa, pb, loss="bogus")

    def test_too_few_observations_raises(self) -> None:
        with pytest.raises(ValueError):
            diebold_mariano([1.0, 2.0], [1.1, 1.9], [0.9, 2.1])


class TestPairedBootstrap:
    def test_ci_excludes_zero_when_b_is_better(self) -> None:
        y, pa, pb = _synthetic(seed=5, sigma_a=1.5, sigma_b=0.4)
        ci = paired_block_bootstrap(y, pa, pb, metric="rmse", n_resamples=2_000, seed=7)
        assert ci.delta < 0
        assert ci.ci_high < 0
        assert ci.ci_low <= ci.ci_high
        assert ci.n_resamples == 2_000

    def test_ci_includes_zero_for_equal_models(self) -> None:
        y, pa, _ = _synthetic(seed=6)
        rng = np.random.default_rng(11)
        pb = pa + rng.normal(0.0, 1e-6, pa.size)
        ci = paired_block_bootstrap(y, pa, pb, metric="rmse", n_resamples=2_000, seed=8)
        assert ci.ci_low <= 0 <= ci.ci_high

    def test_mae_metric_supported(self) -> None:
        y, pa, pb = _synthetic(seed=7)
        ci = paired_block_bootstrap(y, pa, pb, metric="mae", n_resamples=1_000, seed=2)
        assert ci.delta < 0


class TestCohensD:
    def test_strong_effect(self) -> None:
        a = np.array([1.0, 1.05, 0.98, 1.02, 1.01])
        b = np.array([0.50, 0.52, 0.48, 0.51, 0.49])
        assert cohens_d(a, b) < -2.0

    def test_zero_difference(self) -> None:
        a = np.array([1.0, 1.0, 1.0, 1.0])
        assert cohens_d(a, a) == 0.0


class TestHolm:
    def test_orders_and_rejects(self) -> None:
        adjusted = holm_bonferroni([0.001, 0.04, 0.5], alpha=0.05)
        assert len(adjusted) == 3
        assert adjusted[0][2] is True
        assert adjusted[2][2] is False
        for raw, adj, _ in adjusted:
            assert adj >= raw - 1e-12
            assert 0.0 <= adj <= 1.0

    def test_monotone_after_correction(self) -> None:
        adjusted = holm_bonferroni([0.01, 0.02, 0.03, 0.5], alpha=0.05)
        adj = [a for _, a, _ in adjusted]
        order = sorted(range(len(adj)), key=lambda i: [0.01, 0.02, 0.03, 0.5][i])
        sorted_adj = [adj[i] for i in order]
        for i in range(1, len(sorted_adj)):
            assert sorted_adj[i] >= sorted_adj[i - 1] - 1e-12

    def test_empty_input(self) -> None:
        assert holm_bonferroni([]) == []
