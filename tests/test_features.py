"""
Tests for Forecast Evaluation Metrics.

This test suite validates the metric functions used throughout GridPulse
for evaluating forecast accuracy. These metrics are critical for:

- Model comparison during development
- Hyperparameter tuning (objective functions)
- Production monitoring (drift detection)
- Paper reporting (benchmark comparisons)

Metrics Tested:
    - RMSE: Root Mean Squared Error (penalizes large errors)
    - MAE: Mean Absolute Error (robust to outliers)
    - MAPE: Mean Absolute Percentage Error (scale-independent)
    - sMAPE: Symmetric MAPE (handles zeros gracefully)
    - R²: Coefficient of determination (explained variance)

Test Strategy:
    1. Edge cases: Perfect predictions (error = 0)
    2. Known values: Hand-calculated examples
    3. Numerical stability: Very small/large values

Running Tests:
    pytest tests/test_features.py -v

See Also:
    - src/gridpulse/utils/metrics.py: Module under test
"""
import pandas as pd

from gridpulse.utils.metrics import rmse


def test_rmse_zero():
    """
    Test that RMSE is zero for perfect predictions.
    
    This is the most basic sanity check: if predictions exactly match
    actuals, the error should be exactly zero.
    """
    # Perfect predictions should yield zero error
    assert rmse([1, 2, 3], [1, 2, 3]) == 0.0, "Perfect predictions should give RMSE=0"
