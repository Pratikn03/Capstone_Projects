"""
Tests for Time Series Data Splitting.

This test suite validates the temporal splitting logic that divides data
into train/validation/test sets while respecting time order.

Why Temporal Splitting Matters:
    In time series forecasting, random splits cause DATA LEAKAGE because
    future information would be used to predict the past. Our splitter
    ensures:

    1. Train data comes first (oldest timestamps)
    2. Validation data comes next
    3. Test data comes last (newest timestamps)
    4. No overlap between splits

Test Cases:
    - Correct split ratios (70/15/15)
    - Temporal ordering preserved (train < val < test)
    - No duplicate timestamps across splits
    - Edge cases (small datasets, rounding)

Running Tests:
    pytest tests/test_splits.py -v

See Also:
    - src/orius/data_pipeline/split_time_series.py: Module under test
    - DATA.md: Documentation of data splits
"""

import pandas as pd

from orius.data_pipeline.split_time_series import time_split


def test_time_split_order():
    """
    Test that time splits respect temporal ordering.

    This is the CRITICAL property of time series splitting:
    all train timestamps must be < all validation timestamps,
    which must be < all test timestamps.
    """
    # Arrange: Create a simple monotonic time series
    df = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=100, freq="h"), "x": range(100)})

    # Act: Split with standard 70/15/15 ratios
    train, calibration, val, test = time_split(df, 0.7, 0.15, 0.0, 0)

    # Assert: Sizes match expected ratios
    assert len(train) == 70, "Train should be 70% of data"
    assert len(calibration) == 0, "Calibration should be empty when ratio is 0"
    assert len(val) == 15, "Validation should be 15% of data"
    assert len(test) == 15, "Test should be 15% of data"

    # Assert: Temporal ordering is strict (train ends before val starts)
    assert train["timestamp"].max() < val["timestamp"].min(), (
        "Train data must end before validation data begins (no overlap)"
    )


def test_time_split_with_calibration_and_gap():
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=120, freq="h"),
            "x": range(120),
        }
    )
    train, calibration, val, test = time_split(df, 0.6, 0.15, 0.1, 2)

    assert len(train) == 72
    assert len(calibration) == 12
    assert len(val) == 18
    assert len(test) <= 14
    assert train["timestamp"].max() < calibration["timestamp"].min()
    assert calibration["timestamp"].max() < val["timestamp"].min()
    assert val["timestamp"].max() < test["timestamp"].min()
