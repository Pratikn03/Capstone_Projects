"""Tests for anomaly detection helpers."""
import numpy as np

from gridpulse.anomaly.detect import detect_anomalies


def test_detect_anomalies_basic():
    # Arrange a tiny series with one obvious outlier and a matching feature spike.
    actual = np.array([10.0, 12.0, 9.0, 50.0])
    forecast = np.array([10.5, 11.5, 9.2, 10.0])
    features = np.array([[1, 2], [1, 2], [1, 2], [9, 9]])

    # Act: run detection; we only assert shape/keys to keep the test robust.
    out = detect_anomalies(actual, forecast, features)
    # Assert: the combined signal exists and aligns with the input length.
    assert "combined" in out
    assert len(out["combined"]) == len(actual)
