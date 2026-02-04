"""Forecasting datasets for sequence models."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
from torch.utils.data import Dataset

@dataclass
class SeqConfig:
    """Window configuration for sequence models."""
    lookback: int = 168
    horizon: int = 24

class TimeSeriesWindowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, cfg: SeqConfig):
        """Create a sliding-window dataset for (lookback -> horizon) forecasting."""
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
        self.cfg = cfg
        self.n = len(X)

        # last index for a full window + horizon target
        self.max_i = self.n - (cfg.lookback + cfg.horizon)

    def __len__(self):
        """Number of valid windows."""
        return max(0, self.max_i)

    def __getitem__(self, idx):
        """Return a single window (X history) and target (future horizon)."""
        lb = self.cfg.lookback
        hz = self.cfg.horizon
        x = self.X[idx:idx+lb]
        # predict horizon steps ahead: we use the *y* sequence for next hz and train with mean loss
        y = self.y[idx+lb:idx+lb+hz]
        return torch.from_numpy(x), torch.from_numpy(y)
