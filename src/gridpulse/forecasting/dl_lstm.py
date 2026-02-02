"""Forecasting: dl lstm."""
from __future__ import annotations
import torch
import torch.nn as nn

class LSTMForecaster(nn.Module):
    def __init__(
        # Key: prepare features/targets and train or evaluate models
        self, 
        n_features: int, 
        hidden_size: int = 128, 
        num_layers: int = 2, 
        dropout: float = 0.1,
        horizon: int = 24
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        # Project last hidden state to the full horizon
        self.head = nn.Linear(hidden_size, horizon)

    def forward(self, x):
        # x: (B, T, F)
        out, _ = self.lstm(x)
        # Take the output of the last time step
        last_step = out[:, -1, :]  # (B, H)
        y = self.head(last_step)   # (B, Horizon)
        return y
