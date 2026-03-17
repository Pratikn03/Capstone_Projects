"""Comprehensive tests for DC3S calibration / uncertainty inflation."""
from __future__ import annotations

import numpy as np
import pytest

from orius.dc3s.calibration import (
    build_uncertainty_set,
    build_uncertainty_set_kappa,
    calibrate_ambiguity_lambda,
    inflate_interval,
    inflate_q,
)


class TestInflateQ:
    def test_scalar(self):
        assert np.isclose(inflate_q(10.0, 2.0)[0], 20.0)

    def test_array(self):
        out = inflate_q([5.0, 10.0], 1.5)
        np.testing.assert_allclose(out, [7.5, 15.0])

    def test_identity_at_one(self):
        out = inflate_q([3.0, 7.0], 1.0)
        np.testing.assert_allclose(out, [3.0, 7.0])


class TestInflateInterval:
    def test_symmetric_expansion(self):
        lo, hi = inflate_interval(100.0, 200.0, 2.0)
        assert np.isclose(lo[0], 50.0)
        assert np.isclose(hi[0], 250.0)

    def test_identity_at_one(self):
        lo, hi = inflate_interval(90.0, 110.0, 1.0)
        assert np.isclose(lo[0], 90.0)
        assert np.isclose(hi[0], 110.0)

    def test_lower_gt_upper_raises(self):
        with pytest.raises(ValueError, match="lower cannot exceed upper"):
            inflate_interval(110.0, 90.0, 1.5)

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="same length"):
            inflate_interval([1.0, 2.0], [3.0], 1.0)

    def test_array_inputs(self):
        lo, hi = inflate_interval([80.0, 90.0], [120.0, 130.0], 1.5)
        assert lo.shape == (2,)
        assert hi.shape == (2,)


class TestCalibrateAmbiguityLambda:
    def test_quantile_computation(self):
        lam = calibrate_ambiguity_lambda([1.0, 2.0, 3.0, 4.0, 5.0], quantile=0.5, scale=1.0)
        assert lam == pytest.approx(3.0, abs=0.5)

    def test_min_lambda_floor(self):
        lam = calibrate_ambiguity_lambda([0.1, 0.2], quantile=0.5, scale=1.0, min_lambda=5.0)
        assert lam >= 5.0

    def test_max_lambda_cap(self):
        lam = calibrate_ambiguity_lambda([100.0, 200.0], quantile=0.9, scale=2.0, max_lambda=10.0)
        assert lam <= 10.0

    def test_scale_factor(self):
        lam1 = calibrate_ambiguity_lambda([5.0], quantile=0.5, scale=1.0)
        lam2 = calibrate_ambiguity_lambda([5.0], quantile=0.5, scale=2.0)
        assert lam2 == pytest.approx(2.0 * lam1)


class TestBuildUncertaintySet:
    def test_no_inflation_at_perfect_telemetry(self):
        cfg = {"k_q": 0.8, "k_drift": 0.6, "infl_max": 3.0}
        lo, hi, meta = build_uncertainty_set(100.0, 10.0, 1.0, False, cfg)
        assert meta["inflation"] == pytest.approx(1.0)
        assert np.isclose(lo[0], 90.0)
        assert np.isclose(hi[0], 110.0)

    def test_full_inflation_at_degraded(self):
        cfg = {"k_q": 0.8, "k_drift": 0.6, "infl_max": 3.0}
        lo, hi, meta = build_uncertainty_set(100.0, 10.0, 0.5, True, cfg)
        expected_infl = 1.0 + 0.8 * 0.5 + 0.6
        assert meta["inflation"] == pytest.approx(expected_infl)

    def test_infl_max_clipping(self):
        cfg = {"k_q": 10.0, "k_drift": 10.0, "infl_max": 2.5}
        _, _, meta = build_uncertainty_set(100.0, 10.0, 0.0, True, cfg)
        assert meta["inflation"] == pytest.approx(2.5)

    def test_cooldown_smoothing(self):
        cfg = {"k_q": 0.8, "k_drift": 0.6, "infl_max": 3.0, "cooldown_smoothing": 0.5}
        _, _, meta = build_uncertainty_set(100.0, 10.0, 0.5, True, cfg, prev_inflation=1.0)
        raw = 1.0 + 0.8 * 0.5 + 0.6
        smoothed = 0.5 * 1.0 + 0.5 * raw
        assert meta["inflation"] == pytest.approx(smoothed, abs=0.01)

    def test_base_lower_upper_variant(self):
        cfg = {"k_q": 0.8, "k_drift": 0.6, "infl_max": 3.0}
        lo, hi, meta = build_uncertainty_set(
            100.0, 10.0, 1.0, False, cfg, base_lower=85.0, base_upper=115.0
        )
        assert lo.shape[0] == 1
        assert hi.shape[0] == 1

    def test_ambiguity_widening(self):
        cfg = {
            "k_q": 0.0, "k_drift": 0.0, "infl_max": 3.0,
            "ambiguity": {"lambda_mw": 10.0, "min_w": 0.05, "max_extra": 1.0},
        }
        lo, hi, meta = build_uncertainty_set(100.0, 10.0, 0.5, False, cfg)
        assert lo[0] < 90.0
        assert hi[0] > 110.0

    def test_sensitivity_component(self):
        cfg = {"k_q": 0.0, "k_drift": 0.0, "k_sensitivity": 1.0, "infl_max": 5.0, "sensitivity_t": 1.0}
        _, _, meta = build_uncertainty_set(100.0, 10.0, 1.0, False, cfg)
        assert meta["inflation"] > 1.0

    def test_ftit_ro_law(self):
        cfg = {
            "law": "ftit_ro",
            "reliability": {"min_w": 0.05},
            "ftit": {"delta": 0.05, "sigma2_floor": 1e-6, "sigma2_init": 1.0},
            "ftit_runtime": {"sigma2": 4.0},
        }
        _, _, meta = build_uncertainty_set(100.0, 10.0, 0.5, False, cfg)
        assert meta["inflation_rule"] == "ftit_ro"

    def test_meta_fields_all_present(self):
        cfg = {"k_q": 0.8, "k_drift": 0.6, "infl_max": 3.0}
        _, _, meta = build_uncertainty_set(100.0, 10.0, 0.8, False, cfg)
        required = {"w_t_raw", "w_t_used", "inflation", "inflation_raw", "k_quality", "k_drift",
                     "drift_flag", "infl_max", "interval_width", "inflation_rule"}
        assert required <= set(meta.keys())

    def test_array_yhat_and_q(self):
        cfg = {"k_q": 0.8, "k_drift": 0.6, "infl_max": 3.0}
        lo, hi, _ = build_uncertainty_set([100.0, 200.0], [10.0, 20.0], 1.0, False, cfg)
        assert lo.shape == (2,)
        assert hi.shape == (2,)

    def test_broadcast_scalar_q(self):
        cfg = {"k_q": 0.0, "k_drift": 0.0, "infl_max": 3.0}
        lo, hi, _ = build_uncertainty_set([100.0, 200.0], 10.0, 1.0, False, cfg)
        assert lo.shape == (2,)


class TestBuildUncertaintySetKappa:
    def test_kappa_formula(self):
        cfg = {"reliability": {"min_w": 0.05}, "infl_max": 5.0}
        lo, hi, meta = build_uncertainty_set_kappa(
            100.0, 10.0, 0.5, False, cfg, sigma_sq=4.0, delta=0.05
        )
        assert meta["inflation_law"] == "kappa"
        assert meta["kappa"] > 1.0

    def test_drift_penalty_reduces_w(self):
        cfg = {"reliability": {"min_w": 0.05}, "infl_max": 5.0, "kappa_drift_penalty": 0.5}
        _, _, meta_no_drift = build_uncertainty_set_kappa(100.0, 10.0, 0.8, False, cfg, sigma_sq=1.0)
        _, _, meta_drift = build_uncertainty_set_kappa(100.0, 10.0, 0.8, True, cfg, sigma_sq=1.0)
        assert meta_drift["w_t_used"] < meta_no_drift["w_t_used"]

    def test_smoothing_with_prev_inflation(self):
        cfg = {"reliability": {"min_w": 0.05}, "infl_max": 5.0, "cooldown_smoothing": 0.5}
        _, _, meta = build_uncertainty_set_kappa(
            100.0, 10.0, 0.5, False, cfg, sigma_sq=4.0, prev_inflation=1.0
        )
        assert meta["kappa"] >= 1.0
