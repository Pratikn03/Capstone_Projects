import numpy as np
import pytest
from orius.dc3s.calibration import (
    build_uncertainty_set,
    calibrate_ambiguity_lambda,
    inflate_interval,
    inflate_q,
)


def test_inflate_q():
    q_out = inflate_q(10.0, 1.5)
    assert np.isclose(q_out[0], 15.0)


def test_inflate_interval():
    lo, hi = inflate_interval(100.0, 200.0, 1.5)
    # original interval: mid=150.0, half=50.0
    # new half width: 50.0 * 1.5 = 75.0
    # new lo = 150.0 - 75.0 = 75.0
    # new hi = 150.0 + 75.0 = 225.0
    assert np.isclose(lo[0], 75.0)
    assert np.isclose(hi[0], 225.0)


def test_calibrate_ambiguity_lambda():
    residuals = [2.0, 4.0, 1.0, 8.0, 5.0]
    # Sorted: 1, 2, 4, 5, 8
    # 80th percentile using linear interpolation: index 3.2 --> 5 + 0.2*(8-5) = 5.6
    lam = calibrate_ambiguity_lambda(residuals, quantile=0.8, scale=2.0)
    # lambda = 5.6 * 2.0 = 11.2
    assert np.isclose(lam, 11.2)


def test_build_uncertainty_set_no_inflation():
    cfg = {"k_q": 0.8, "k_drift": 0.6, "infl_max": 3.0}
    # Quality w_t=1.0, no drift -> inflation = 1.0
    lo, hi, meta = build_uncertainty_set(yhat=100.0, q=10.0, w_t=1.0, drift_flag=False, cfg=cfg)
    assert np.isclose(lo[0], 90.0)
    assert np.isclose(hi[0], 110.0)
    assert meta["inflation"] == 1.0


def test_build_uncertainty_set_drift_inflation():
    cfg = {"k_q": 0.8, "k_drift": 0.6, "infl_max": 3.0}
    # Quality w_t=0.5, drift_flag=True
    # infl = 1.0 + 0.8*(1.0-0.5) + 0.6*1 = 1.0 + 0.4 + 0.6 = 2.0
    lo, hi, meta = build_uncertainty_set(yhat=100.0, q=10.0, w_t=0.5, drift_flag=True, cfg=cfg)
    assert np.isclose(lo[0], 80.0)
    assert np.isclose(hi[0], 120.0)
    assert meta["inflation"] == 2.0


def test_build_uncertainty_set_clipped():
    cfg = {"k_q": 5.0, "k_drift": 5.0, "infl_max": 3.0} # massive constants to force clipping
    # Quality w_t=0.0, drift_flag=True
    lo, hi, meta = build_uncertainty_set(yhat=100.0, q=10.0, w_t=0.0, drift_flag=True, cfg=cfg)
    # Max inflation is 3.0
    assert np.isclose(lo[0], 70.0)
    assert np.isclose(hi[0], 130.0)
    assert meta["inflation"] == 3.0


def test_build_uncertainty_set_ftit_ro_formula():
    cfg = {
        "law": "ftit_ro",
        "reliability": {"min_w": 0.05},
        "ftit": {"delta": 0.05, "eps_interval": 1.0e-6, "sigma2_floor": 1.0e-6, "sigma2_init": 1.0},
        "ftit_runtime": {"sigma2": 4.0},
    }
    lo, hi, meta = build_uncertainty_set(yhat=100.0, q=10.0, w_t=0.5, drift_flag=False, cfg=cfg)
    expected_kappa = 1.0 + np.sqrt(2.0 * 4.0 * np.log(1.0 / (0.05 * 0.5))) / 20.0
    assert np.isclose(meta["inflation"], expected_kappa)
    assert np.isclose(lo[0], 100.0 - 10.0 * expected_kappa)
    assert np.isclose(hi[0], 100.0 + 10.0 * expected_kappa)


def test_build_uncertainty_set_ftit_ro_meta_fields():
    cfg = {
        "law": "ftit_ro",
        "reliability": {"min_w": 0.05},
        "ftit": {"delta": 0.10, "eps_interval": 1.0e-6, "sigma2_floor": 1.0e-6, "sigma2_init": 1.0},
        "ftit_runtime": {"sigma2": 2.5},
    }
    _, _, meta = build_uncertainty_set(yhat=120.0, q=8.0, w_t=0.4, drift_flag=True, cfg=cfg)
    assert meta["inflation_rule"] == "ftit_ro"
    assert meta["delta"] == pytest.approx(0.10)
    assert meta["sigma2"] == pytest.approx(2.5)
    assert meta["w_used"] == pytest.approx(0.4)
    assert meta["q_multiplier"] == pytest.approx(1.0)
