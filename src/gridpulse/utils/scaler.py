"""
Utilities: Feature Scaling for Machine Learning.

This module provides a simple StandardScaler implementation for normalizing
features before feeding them to ML models. Standardization (z-score normalization)
is important because:

1. **GBM models**: While tree-based models are scale-invariant, regularization
   parameters (L1/L2) work better with standardized features.
   
2. **Neural networks**: LSTM/TCN models are highly sensitive to input scale.
   Standardization ensures stable gradient flow and faster convergence.
   
3. **Interpretability**: Standardized feature importances are more comparable.

The StandardScaler uses the formula:
    X_scaled = (X - mean) / std
    
Why not use sklearn's StandardScaler?
    - Simpler serialization for model bundles
    - No sklearn dependency for inference-only deployments
    - Full control over edge cases (zero variance, etc.)

Usage:
    >>> from gridpulse.utils.scaler import StandardScaler
    >>> scaler = StandardScaler.fit(X_train)
    >>> X_train_scaled = scaler.transform(X_train)
    >>> X_test_scaled = scaler.transform(X_test)  # Uses train statistics!
    
Note:
    Always fit on training data only to avoid data leakage.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class StandardScaler:
    """
    Standard scaler for feature normalization (z-score standardization).
    
    This class computes mean and standard deviation from training data,
    then applies the transformation: X_scaled = (X - mean) / std
    
    Attributes:
        mean: Per-feature means computed from fit()
        std: Per-feature standard deviations from fit()
    """
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, X: np.ndarray) -> "StandardScaler":
        """Fit a scaler using mean and std from data."""
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        std = np.where(std < 1e-12, 1.0, std)
        return cls(mean=mean, std=std)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Standardize data using fitted mean/std."""
        return (X - self.mean) / self.std

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Undo standardization."""
        return X * self.std + self.mean

    def to_dict(self) -> dict:
        """Serialize scaler parameters for saving in model bundles."""
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict | None) -> "StandardScaler" | None:
        """Restore scaler from a serialized dict."""
        if not payload:
            return None
        mean = np.asarray(payload.get("mean"), dtype=float)
        std = np.asarray(payload.get("std"), dtype=float)
        if mean.size == 0 or std.size == 0:
            return None
        return cls(mean=mean, std=std)
