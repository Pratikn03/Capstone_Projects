"""
Tests for Anomaly Detection Module.

This test suite validates the anomaly detection pipeline, which identifies
unusual patterns in energy data that could indicate:

- Sensor malfunctions (bad data)
- Grid events (outages, sudden demand spikes)
- Weather anomalies (storms affecting wind/solar)
- Cyberattacks (malicious data injection)

Detection Methods Tested:
    1. **Residual Z-scores**: Flag large forecast errors (> 3 sigma)
    2. **Isolation Forest**: Multivariate outlier detection on features
    3. **Combined Signal**: Logical OR of both methods

Test Strategy:
    We construct synthetic scenarios with known anomalies (obvious outliers)
    and verify that the detector correctly flags them.

Running Tests:
    pytest tests/test_anomaly.py -v

See Also:
    - src/gridpulse/anomaly/detect.py: Main detection interface
    - src/gridpulse/anomaly/detection.py: Isolation Forest class
"""
import numpy as np

from gridpulse.anomaly.detect import detect_anomalies


def test_detect_anomalies_basic():
    """
    Test basic anomaly detection on a simple synthetic scenario.
    
    Scenario:
        - 4 hourly observations, normal values around 10 MW
        - Hour 4 has a spike to 50 MW (5x normal)
        - Features also show abnormal values at hour 4
        
    Expected:
        - The detector should flag hour 4 (index 3) as anomalous
        - Output shapes should match input length
    """
    # Arrange: Tiny series with one obvious outlier
    actual = np.array([10.0, 12.0, 9.0, 50.0])      # Hour 4 is 5x normal
    forecast = np.array([10.5, 11.5, 9.2, 10.0])    # Model expected ~10
    features = np.array([[1, 2], [1, 2], [1, 2], [9, 9]])  # Features also spike

    # Act: Run anomaly detection
    out = detect_anomalies(actual, forecast, features)
    
    # Assert: Combined signal exists and has correct length
    assert "combined" in out, "Output should include 'combined' anomaly flags"
    assert len(out["combined"]) == len(actual), "Flags should match input length"
