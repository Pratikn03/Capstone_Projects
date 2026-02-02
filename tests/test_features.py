"""Tests for test features."""
import pandas as pd
from gridpulse.utils.metrics import rmse

def test_rmse_zero():
    # Key: test setup and assertions
    assert rmse([1,2,3],[1,2,3]) == 0.0
