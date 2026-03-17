"""
Tests for Conformal Prediction Intervals.

This test suite validates the conformal prediction module, which provides
uncertainty quantification for point forecasts. Conformal prediction is
a distribution-free method that provides valid coverage guarantees.

Key Properties Being Tested:
    1. **Coverage Validity**: Intervals contain the true value at least
       (1-alpha) fraction of the time (e.g., 90% for alpha=0.10)
    2. **Shape Correctness**: Output intervals match input dimensions
    3. **Adaptivity**: Interval widths adjust to local prediction difficulty

Test Strategy:
    We use synthetic data where we know the true distribution, allowing us
    to verify that coverage is approximately correct. Real data tests are
    in the integration test suite.

Running Tests:
    pytest tests/test_conformal.py -v

See Also:
    - src/orius/forecasting/uncertainty/conformal.py: Module under test
    - docs/EVALUATION.md: Discussion of uncertainty quantification
"""
import numpy as np
import pytest

from orius.forecasting.uncertainty.conformal import AdaptiveConformal, ConformalConfig, ConformalInterval


def test_conformal_horizon_wise():
    """
    Test horizon-wise conformal prediction intervals.
    
    Horizon-wise calibration computes separate quantiles for each forecast
    step. This is important because error distributions often vary by horizon
    (e.g., hour 24 has wider intervals than hour 1).
    
    Validation:
        - Shapes are correct (N, horizon) for bounds
        - Empirical coverage is approximately (1-alpha)
    """
    # Create reproducible synthetic data
    rng = np.random.default_rng(0)
    
    # Ground truth: 1000 samples, 24-hour horizon
    y_true = rng.normal(size=(1000, 24))
    
    # Predictions: true + noise (simulates forecast errors)
    y_pred = y_true + rng.normal(scale=0.5, size=(1000, 24))

    # Configure 90% prediction intervals with horizon-wise calibration
    ci = ConformalInterval(ConformalConfig(alpha=0.10, horizon_wise=True, rolling=False))
    ci.fit_calibration(y_true, y_pred)

    # Generate intervals for a subset
    lo, hi = ci.predict_interval(y_pred[:50])
    
    # Assert: bounds have correct shape
    assert lo.shape == (50, 24), "Lower bound shape should match (n_samples, horizon)"
    
    # Assert: coverage is approximately 90% (allow some variance)
    cov = ci.coverage(y_true[:50], y_pred[:50])
    assert 0.75 <= cov <= 0.99, f"Coverage {cov:.2f} should be close to 90%"


def test_adaptive_global_alpha_updates():
    faci = AdaptiveConformal(alpha=0.10, gamma=0.05, mode="global")
    interval = (np.array([0.0]), np.array([2.0]))

    faci.update(y_true=np.array([3.0]), y_pred_interval=interval)
    assert np.isclose(float(faci.alpha_t), 0.15)

    faci.update(y_true=np.array([1.0]), y_pred_interval=interval)
    assert np.isclose(float(faci.alpha_t), 0.10)


def test_adaptive_global_width_reacts_immediately():
    faci = AdaptiveConformal(alpha=0.10, gamma=0.05, mode="global")
    interval = (np.array([0.0, 0.0]), np.array([2.0, 2.0]))

    lo_wide, hi_wide = faci.update(y_true=np.array([3.0, 1.0]), y_pred_interval=interval)
    width_wide = hi_wide - lo_wide
    assert np.allclose(width_wide, np.array([3.0, 3.0]))

    lo_tight, hi_tight = faci.update(y_true=np.array([1.0, 1.0]), y_pred_interval=interval)
    width_tight = hi_tight - lo_tight
    assert np.all(width_tight < width_wide)


def test_adaptive_horizon_wise_independent_updates():
    faci = AdaptiveConformal(alpha=0.10, gamma=0.05, mode="horizon_wise")
    interval = (np.array([0.0, 0.0, 0.0]), np.array([2.0, 2.0, 2.0]))

    lo_new, hi_new = faci.update(y_true=np.array([3.0, 1.0, 3.0]), y_pred_interval=interval)
    assert isinstance(faci.alpha_t, np.ndarray)
    assert np.allclose(faci.alpha_t, np.array([0.15, 0.05, 0.15]))
    assert np.allclose(hi_new - lo_new, np.array([3.0, 1.0, 3.0]))


def test_adaptive_alpha_clamping():
    faci = AdaptiveConformal(alpha=0.95, gamma=0.10, mode="global", alpha_min=0.01, alpha_max=0.99)
    interval = (np.array([0.0]), np.array([1.0]))

    faci.update(y_true=np.array([2.0]), y_pred_interval=interval)
    assert np.isclose(float(faci.alpha_t), 0.99)

    for _ in range(30):
        faci.update(y_true=np.array([0.5]), y_pred_interval=interval)
    assert np.isclose(float(faci.alpha_t), 0.01)


def test_adaptive_interval_validation():
    faci = AdaptiveConformal()
    with pytest.raises(ValueError, match="Interval lower bound must be <= upper bound"):
        faci.update(y_true=[1.0, 2.0], y_pred_interval=([0.0, 3.0], [0.5, 2.5]))

    with pytest.raises(ValueError, match="broadcast-compatible"):
        faci.update(y_true=[1.0, 2.0], y_pred_interval=([0.0, 1.0, 2.0], [1.0, 2.0, 3.0]))


def test_existing_conformal_interval_unchanged():
    rng = np.random.default_rng(1)
    y_true = rng.normal(size=(200, 8))
    y_pred = y_true + rng.normal(scale=0.4, size=(200, 8))

    ci = ConformalInterval(ConformalConfig(alpha=0.10, horizon_wise=True, rolling=False))
    ci.fit_calibration(y_true, y_pred)
    lo, hi = ci.predict_interval(y_pred[:20])

    assert lo.shape == (20, 8)
    assert hi.shape == (20, 8)
    cov = ci.coverage(y_true[:20], y_pred[:20])
    assert 0.70 <= cov <= 1.0


def test_cqr_calibration_and_interval_evaluation():
    rng = np.random.default_rng(7)
    y_true_cal = rng.normal(loc=50.0, scale=5.0, size=(240, 4))
    q_lo_cal = y_true_cal - np.abs(rng.normal(loc=3.0, scale=0.5, size=(240, 4)))
    q_hi_cal = y_true_cal + np.abs(rng.normal(loc=3.0, scale=0.5, size=(240, 4)))

    y_true_test = rng.normal(loc=52.0, scale=5.5, size=(120, 4))
    q_lo_test = y_true_test - np.abs(rng.normal(loc=2.8, scale=0.6, size=(120, 4)))
    q_hi_test = y_true_test + np.abs(rng.normal(loc=2.8, scale=0.6, size=(120, 4)))

    ci = ConformalInterval(ConformalConfig(alpha=0.10, method="cqr", horizon_wise=True, rolling=False))
    ci.fit_calibration_cqr(y_true_cal, q_lo_cal, q_hi_cal)

    lo, hi = ci.predict_interval_cqr(q_lo_test, q_hi_test)
    assert lo.shape == q_lo_test.shape
    assert hi.shape == q_hi_test.shape
    assert np.all(lo <= hi)

    eval_metrics = ci.evaluate_intervals_cqr(y_true_test, q_lo_test, q_hi_test, per_horizon=True)
    assert "global_coverage" in eval_metrics
    assert "global_mean_width" in eval_metrics
    assert "per_horizon_picp" in eval_metrics
    assert "per_horizon_mpiw" in eval_metrics
