"""Utilities: scaler."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class StandardScaler:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, X: np.ndarray) -> "StandardScaler":
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        std = np.where(std < 1e-12, 1.0, std)
        return cls(mean=mean, std=std)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return X * self.std + self.mean

    def to_dict(self) -> dict:
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict | None) -> "StandardScaler" | None:
        if not payload:
            return None
        mean = np.asarray(payload.get("mean"), dtype=float)
        std = np.asarray(payload.get("std"), dtype=float)
        if mean.size == 0 or std.size == 0:
            return None
        return cls(mean=mean, std=std)
