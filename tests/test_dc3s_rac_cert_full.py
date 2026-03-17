"""Comprehensive tests for DC3S RAC-Cert module."""
from __future__ import annotations

import json

import numpy as np
import pytest

from orius.dc3s.rac_cert import (
    RACCertConfig,
    RACCertModel,
    compute_dispatch_sensitivity,
    compute_inflation,
    compute_q_multiplier,
    normalize_sensitivity,
)


def _synthetic_calibration(n=200, seed=42):
    rng = np.random.default_rng(seed)
    y = np.cumsum(rng.normal(0, 1, n)) + 100.0
    noise = rng.normal(0, 2, n)
    q_lo = y - 5.0 + noise * 0.3
    q_hi = y + 5.0 + noise * 0.3
    return y, q_lo, q_hi


class TestNormalizeSensitivity:
    def test_clamps_to_unit(self):
        assert normalize_sensitivity(10.0, 0.5) == pytest.approx(1.0)

    def test_zero(self):
        assert normalize_sensitivity(0.0, 0.5) == pytest.approx(0.0)

    def test_linear_scaling(self):
        assert normalize_sensitivity(0.25, 0.5) == pytest.approx(0.5)

    def test_negative_clamped(self):
        assert normalize_sensitivity(-1.0, 0.5) == pytest.approx(0.0)


class TestComputeQMultiplier:
    def test_identity_at_perfect(self):
        cfg = RACCertConfig(beta_reliability=0.7, beta_sensitivity=0.5)
        q, meta = compute_q_multiplier(w_t=1.0, sensitivity_norm=0.0, cfg=cfg)
        assert q == pytest.approx(1.0)

    def test_increases_with_lower_w(self):
        cfg = RACCertConfig(beta_reliability=0.7, beta_sensitivity=0.0)
        q1, _ = compute_q_multiplier(w_t=0.9, sensitivity_norm=0.0, cfg=cfg)
        q2, _ = compute_q_multiplier(w_t=0.3, sensitivity_norm=0.0, cfg=cfg)
        assert q2 > q1

    def test_clamped_by_max(self):
        cfg = RACCertConfig(beta_reliability=10.0, max_q_multiplier=2.0)
        q, _ = compute_q_multiplier(w_t=0.0, sensitivity_norm=0.0, cfg=cfg)
        assert q == pytest.approx(2.0)

    def test_sensitivity_effect(self):
        cfg = RACCertConfig(beta_reliability=0.0, beta_sensitivity=1.0)
        q, _ = compute_q_multiplier(w_t=1.0, sensitivity_norm=0.5, cfg=cfg)
        assert q > 1.0


class TestComputeInflation:
    def test_identity_at_perfect(self):
        cfg = RACCertConfig()
        infl, _ = compute_inflation(w_t=1.0, drift_flag=False, sensitivity_norm=0.0,
                                     k_quality=0.8, k_drift=0.6, cfg=cfg)
        assert infl == pytest.approx(1.0)

    def test_components_add_up(self):
        cfg = RACCertConfig(k_sensitivity=0.4, infl_max=10.0)
        infl, meta = compute_inflation(w_t=0.5, drift_flag=True, sensitivity_norm=0.5,
                                        k_quality=0.8, k_drift=0.6, cfg=cfg)
        expected = 1.0 + 0.8 * 0.5 + 0.6 + 0.4 * 0.5
        assert infl == pytest.approx(expected)
        assert meta["quality"] == pytest.approx(0.8 * 0.5)
        assert meta["drift"] == pytest.approx(0.6)

    def test_clipped_by_infl_max(self):
        cfg = RACCertConfig(infl_max=2.0)
        infl, _ = compute_inflation(w_t=0.0, drift_flag=True, sensitivity_norm=1.0,
                                     k_quality=5.0, k_drift=5.0, cfg=cfg)
        assert infl == pytest.approx(2.0)


class TestComputeDispatchSensitivity:
    def test_constant_dispatch_zero_sens(self):
        def probe(load):
            return 5.0, 0.0
        sens = compute_dispatch_sensitivity(load_window=np.array([100.0, 200.0]),
                                             dispatch_probe=probe, sens_eps_mw=25.0)
        assert sens == pytest.approx(0.0)

    def test_varying_dispatch_positive_sens(self):
        def probe(load):
            return 0.0, float(load[0]) * 0.1
        sens = compute_dispatch_sensitivity(load_window=np.array([100.0]),
                                             dispatch_probe=probe, sens_eps_mw=25.0)
        assert sens > 0.0

    def test_empty_window(self):
        sens = compute_dispatch_sensitivity(load_window=np.array([]),
                                             dispatch_probe=lambda x: (0.0, 0.0),
                                             sens_eps_mw=25.0)
        assert sens == 0.0


class TestRACCertModel:
    def test_fit_returns_meta(self):
        y, lo, hi = _synthetic_calibration()
        model = RACCertModel(cfg=RACCertConfig(n_vol_bins=3, vol_window=24))
        meta = model.fit(y, lo, hi)
        assert "global_qhat" in meta
        assert "qhat_by_vol_bin" in meta
        assert len(meta["qhat_by_vol_bin"]) == 3

    def test_qhat_for_context(self):
        y, lo, hi = _synthetic_calibration()
        model = RACCertModel(cfg=RACCertConfig(n_vol_bins=3, vol_window=24))
        model.fit(y, lo, hi)
        qhat = model.qhat_for_context(y[-48:], horizon=24)
        assert qhat.shape == (24,)
        assert np.all(qhat >= 0.0)

    def test_unfitted_raises(self):
        model = RACCertModel(cfg=RACCertConfig())
        with pytest.raises(RuntimeError, match="not fitted"):
            model.assign_context_bins(np.array([1.0, 2.0, 3.0]))

    def test_serialization_round_trip(self):
        y, lo, hi = _synthetic_calibration()
        model = RACCertModel(cfg=RACCertConfig(n_vol_bins=3, vol_window=24))
        model.fit(y, lo, hi)
        s = model.to_json()
        model2 = RACCertModel.from_json(s)
        assert model2.global_qhat == pytest.approx(model.global_qhat)
        np.testing.assert_allclose(model2.qhat_by_vol_bin, model.qhat_by_vol_bin)
        np.testing.assert_allclose(model2.vol_edges, model.vol_edges)

    def test_empty_calibration_raises(self):
        model = RACCertModel(cfg=RACCertConfig())
        with pytest.raises(ValueError, match="non-empty"):
            model.fit(np.array([]), np.array([]), np.array([]))

    def test_mismatched_calibration_raises(self):
        model = RACCertModel(cfg=RACCertConfig())
        with pytest.raises(ValueError, match="same length"):
            model.fit(np.array([1.0, 2.0]), np.array([0.0]), np.array([3.0, 4.0]))

    def test_config_defaults(self):
        cfg = RACCertConfig()
        assert cfg.alpha == 0.10
        assert cfg.n_vol_bins == 3
        assert cfg.infl_max == 2.0

    def test_qhat_for_context_single_bin(self):
        y, lo, hi = _synthetic_calibration()
        model = RACCertModel(cfg=RACCertConfig(n_vol_bins=3, vol_window=24))
        model.fit(y, lo, hi)
        qhat = model.qhat_for_context(np.array([5.0]), horizon=10)
        assert qhat.shape == (10,)
