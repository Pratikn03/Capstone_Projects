from __future__ import annotations
import torch
import torch.nn as nn

class LSTMForecaster(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (B, T, F)
        out, _ = self.lstm(x)
        # predict per-timestep, then map to scalar
        y = self.head(out)  # (B, T, 1)
        return y.squeeze(-1)  # (B, T)
