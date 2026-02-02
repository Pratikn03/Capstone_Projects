"""Tests for test anomaly."""
import numpy as np

from gridpulse.anomaly.detect import detect_anomalies


def test_detect_anomalies_basic():
    # Key: test setup and assertions
    actual = np.array([10.0, 12.0, 9.0, 50.0])
    forecast = np.array([10.5, 11.5, 9.2, 10.0])
    features = np.array([[1, 2], [1, 2], [1, 2], [9, 9]])

    out = detect_anomalies(actual, forecast, features)
    assert "combined" in out
    assert len(out["combined"]) == len(actual)
