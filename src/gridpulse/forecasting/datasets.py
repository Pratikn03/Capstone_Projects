from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
from torch.utils.data import Dataset

@dataclass
class SeqConfig:
    lookback: int = 168
    horizon: int = 24

class TimeSeriesWindowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, cfg: SeqConfig):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
        self.cfg = cfg
        self.n = len(X)

        # last index for a full window + horizon target
        self.max_i = self.n - (cfg.lookback + cfg.horizon)

    def __len__(self):
        return max(0, self.max_i)

    def __getitem__(self, idx):
        lb = self.cfg.lookback
        hz = self.cfg.horizon
        x = self.X[idx:idx+lb]
        # predict horizon steps ahead: we use the *y* sequence for next hz and train with mean loss
        y = self.y[idx+lb:idx+lb+hz]
        return torch.from_numpy(x), torch.from_numpy(y)
