"""Anomaly detection: residual rules."""
from __future__ import annotations
import numpy as np

def residual_zscore_flags(residuals, window: int = 168, z_threshold: float = 3.0):
    residuals = np.asarray(residuals, dtype=float)
    flags = np.zeros_like(residuals, dtype=bool)
    for i in range(window, len(residuals)):
        w = residuals[i-window:i]
        mu = w.mean()
        sd = w.std() + 1e-8
        z = (residuals[i] - mu) / sd
        flags[i] = abs(z) >= z_threshold
    return flags
