"""Comprehensive tests for forecasting conformal prediction."""

from __future__ import annotations

import numpy as np
import pytest

from orius.forecasting.uncertainty.conformal import AdaptiveConformal
from orius.forecasting.uncertainty.cqr import (
    RegimeCQR,
    RegimeCQRConfig,
    assign_bins,
    cqr_scores,
    rolling_volatility,
)


class TestRollingVolatility:
    def test_constant_series_zero_vol(self):
        y = np.ones(50)
        vol = rolling_volatility(y, window=10)
        assert np.allclose(vol, 0.0)

    def test_increasing_window_smooths(self):
        rng = np.random.default_rng(42)
        y = rng.normal(0, 1, 100)
        vol_short = rolling_volatility(y, window=5)
        vol_long = rolling_volatility(y, window=50)
        assert vol_short.std() > vol_long.std() * 0.5

    def test_window_one_returns_zeros(self):
        vol = rolling_volatility(np.array([1.0, 2.0, 3.0]), window=1)
        assert np.allclose(vol, 0.0)

    def test_empty_returns_empty(self):
        vol = rolling_volatility(np.array([]), window=10)
        assert vol.size == 0


class TestAssignBins:
    def test_three_bins(self):
        v = np.linspace(0, 10, 100)
        bins, edges = assign_bins(v, n_bins=3)
        assert bins.shape == (100,)
        assert set(np.unique(bins)) <= {0, 1, 2}

    def test_single_bin(self):
        bins, edges = assign_bins(np.array([1.0, 2.0, 3.0]), n_bins=1)
        assert np.all(bins == 0)

    def test_empty_input(self):
        bins, edges = assign_bins(np.array([]), n_bins=3)
        assert bins.size == 0

    def test_invalid_n_bins_raises(self):
        with pytest.raises(ValueError, match="n_bins"):
            assign_bins(np.array([1.0]), n_bins=0)

    def test_constant_values(self):
        bins, edges = assign_bins(np.ones(20), n_bins=3)
        assert bins.shape == (20,)


class TestCQRScores:
    def test_inside_interval_zero_score(self):
        y = np.array([10.0, 20.0])
        lo = np.array([5.0, 15.0])
        hi = np.array([15.0, 25.0])
        scores = cqr_scores(y, lo, hi)
        np.testing.assert_allclose(scores, 0.0)

    def test_below_interval(self):
        y = np.array([1.0])
        lo = np.array([5.0])
        hi = np.array([10.0])
        scores = cqr_scores(y, lo, hi)
        assert scores[0] == pytest.approx(4.0)

    def test_above_interval(self):
        y = np.array([15.0])
        lo = np.array([5.0])
        hi = np.array([10.0])
        scores = cqr_scores(y, lo, hi)
        assert scores[0] == pytest.approx(5.0)

    def test_mismatched_raises(self):
        with pytest.raises(ValueError, match="same length"):
            cqr_scores(np.array([1.0]), np.array([0.0, 1.0]), np.array([2.0]))


class TestRegimeCQR:
    def _synthetic(self, n=300, seed=42):
        rng = np.random.default_rng(seed)
        y = np.cumsum(rng.normal(0, 1, n)) + 100.0
        q_lo = y - 5.0 + rng.normal(0, 0.5, n)
        q_hi = y + 5.0 + rng.normal(0, 0.5, n)
        return y, q_lo, q_hi

    def test_fit_returns_meta(self):
        y, lo, hi = self._synthetic()
        rcqr = RegimeCQR(cfg=RegimeCQRConfig(n_bins=3, vol_window=24))
        meta = rcqr.fit(y, lo, hi)
        assert "global_qhat" in meta
        assert "qhat_by_bin" in meta
        assert len(meta["qhat_by_bin"]) == 3

    def test_predict_interval_returns_arrays(self):
        y, lo, hi = self._synthetic()
        rcqr = RegimeCQR(cfg=RegimeCQRConfig(n_bins=3, vol_window=24))
        rcqr.fit(y, lo, hi)
        lower, upper, bins = rcqr.predict_interval(y_context=y[-48:], q_lo=lo[-24:], q_hi=hi[-24:])
        assert lower.shape == (24,)
        assert upper.shape == (24,)
        assert np.all(lower <= upper)

    def test_unfitted_raises(self):
        rcqr = RegimeCQR(cfg=RegimeCQRConfig())
        with pytest.raises(RuntimeError, match="not fitted"):
            rcqr.predict_interval(y_context=np.array([1.0]), q_lo=np.array([0.0]), q_hi=np.array([2.0]))

    def test_serialization_round_trip(self):
        y, lo, hi = self._synthetic()
        rcqr = RegimeCQR(cfg=RegimeCQRConfig(n_bins=3, vol_window=24))
        rcqr.fit(y, lo, hi)
        s = rcqr.to_json()
        rcqr2 = RegimeCQR.from_json(s)
        np.testing.assert_allclose(rcqr2.qhat_by_bin, rcqr.qhat_by_bin)
        np.testing.assert_allclose(rcqr2.edges, rcqr.edges)

    def test_empty_cal_raises(self):
        rcqr = RegimeCQR(cfg=RegimeCQRConfig())
        with pytest.raises(ValueError, match="non-empty"):
            rcqr.fit(np.array([]), np.array([]), np.array([]))

    def test_coverage_on_synthetic(self):
        rng = np.random.default_rng(0)
        n = 500
        y = rng.normal(100, 5, n)
        q_lo = y - 6 + rng.normal(0, 0.5, n)
        q_hi = y + 6 + rng.normal(0, 0.5, n)
        rcqr = RegimeCQR(cfg=RegimeCQRConfig(alpha=0.10, n_bins=3, vol_window=24))
        rcqr.fit(y[:300], q_lo[:300], q_hi[:300])
        lo, hi, _ = rcqr.predict_interval(y_context=y[200:400], q_lo=q_lo[300:], q_hi=q_hi[300:])
        coverage = np.mean((y[300:] >= lo) & (y[300:] <= hi))
        assert coverage >= 0.85


class TestAdaptiveConformal:
    def test_init_valid(self):
        ac = AdaptiveConformal(alpha=0.10, gamma=0.05, mode="global")
        assert ac.alpha == 0.10

    def test_init_invalid_mode(self):
        with pytest.raises(ValueError, match="mode"):
            AdaptiveConformal(mode="bad")

    def test_init_negative_gamma(self):
        with pytest.raises(ValueError, match="gamma"):
            AdaptiveConformal(gamma=-1.0)

    def test_init_invalid_alpha_range(self):
        with pytest.raises(ValueError, match="alpha_min must be"):
            AdaptiveConformal(alpha_min=0.99, alpha_max=0.01)

    def test_update_returns_intervals(self):
        ac = AdaptiveConformal(alpha=0.10, gamma=0.05, mode="global")
        y_true = np.array([10.0])
        interval = (np.array([8.0]), np.array([12.0]))
        lo_new, hi_new = ac.update(y_true, interval)
        assert lo_new.shape == (1,)
        assert hi_new.shape == (1,)
        assert lo_new[0] <= hi_new[0]

    def test_update_alpha_stays_in_bounds(self):
        ac = AdaptiveConformal(alpha=0.10, gamma=0.1, alpha_min=0.01, alpha_max=0.99)
        rng = np.random.default_rng(42)
        for _ in range(100):
            y = rng.normal(100, 10, 1)
            ac.update(y, (y - 5, y + 5))
        alpha = ac.alpha_t
        if isinstance(alpha, np.ndarray):
            assert np.all(alpha >= 0.01)
            assert np.all(alpha <= 0.99)
        else:
            assert 0.01 <= alpha <= 0.99

    def test_adaptive_intervals_change(self):
        ac = AdaptiveConformal(alpha=0.10, gamma=0.05, mode="global")
        rng = np.random.default_rng(42)
        lo_prev, hi_prev = np.array([95.0]), np.array([105.0])
        for _ in range(50):
            y = rng.normal(100, 5, 1)
            lo_new, hi_new = ac.update(y, (lo_prev, hi_prev))
            lo_prev, hi_prev = lo_new, hi_new
        assert np.all(np.isfinite(lo_prev))
        assert np.all(np.isfinite(hi_prev))
