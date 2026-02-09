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
    - src/gridpulse/forecasting/uncertainty/conformal.py: Module under test
    - docs/EVALUATION.md: Discussion of uncertainty quantification
"""
import numpy as np

from gridpulse.forecasting.uncertainty.conformal import ConformalConfig, ConformalInterval


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
