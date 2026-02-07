"""Conformal interval tests."""
import numpy as np

from gridpulse.forecasting.uncertainty.conformal import ConformalConfig, ConformalInterval


def test_conformal_horizon_wise():
    rng = np.random.default_rng(0)
    y_true = rng.normal(size=(1000, 24))
    y_pred = y_true + rng.normal(scale=0.5, size=(1000, 24))

    ci = ConformalInterval(ConformalConfig(alpha=0.10, horizon_wise=True, rolling=False))
    ci.fit_calibration(y_true, y_pred)

    lo, hi = ci.predict_interval(y_pred[:50])
    assert lo.shape == (50, 24)
    cov = ci.coverage(y_true[:50], y_pred[:50])
    assert 0.75 <= cov <= 0.99
