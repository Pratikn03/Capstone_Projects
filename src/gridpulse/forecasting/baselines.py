"""Forecasting: baselines."""
from __future__ import annotations
import numpy as np
import pandas as pd

def persistence_24h(df: pd.DataFrame, target: str = "load_mw") -> np.ndarray:
    # Key: prepare features/targets and train or evaluate models
    """Predict next step using value from 24 hours ago (assumes hourly data)."""
    return df[target].shift(24).to_numpy()

def moving_average(df: pd.DataFrame, target: str = "load_mw", window: int = 24) -> np.ndarray:
    return df[target].shift(1).rolling(window).mean().to_numpy()
