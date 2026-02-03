"""Tests for core metric helpers."""
import pandas as pd

from gridpulse.utils.metrics import rmse


def test_rmse_zero():
    # Perfect predictions should yield zero error.
    assert rmse([1, 2, 3], [1, 2, 3]) == 0.0
