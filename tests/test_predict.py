"""Tests for forecasting prediction helpers."""
import numpy as np
import pandas as pd

from gridpulse.forecasting.predict import predict_next_24h


class DummyModel:
    def predict(self, X):
        # Return zeros so the test focuses on wiring/shape, not model skill.
        return np.zeros(len(X))


def test_predict_next_24h_gbm():
    # Arrange a minimal feature frame with a UTC timestamp index.
    df = pd.DataFrame({
        "timestamp": pd.date_range("2020-01-01", periods=48, freq="h", tz="UTC"),
        "feat1": np.random.rand(48),
        "feat2": np.random.rand(48),
    })

    bundle = {
        "model_type": "gbm",
        "model": DummyModel(),
        "feature_cols": ["feat1", "feat2"],
        "target": "load_mw",
        "residual_quantiles": {"0.1": -1.0, "0.9": 1.0},
    }

    # Act: request a 24h forecast from the GBM path.
    out = predict_next_24h(df, bundle, horizon=24)
    # Assert: correct horizon length and quantile outputs exist.
    assert len(out["forecast"]) == 24
    assert "quantiles" in out
