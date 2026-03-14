"""Lightweight N-BEATS style forecaster for multi-feature energy sequences."""
from __future__ import annotations

import torch
import torch.nn as nn


class _NBEATSBlock(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, horizon: int, dropout: float, n_quantiles: int = 1) -> None:
        super().__init__()
        self.horizon = int(horizon)
        self.n_quantiles = max(1, int(n_quantiles))
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.backcast = nn.Linear(hidden_dim, input_dim)
        self.forecast = nn.Linear(hidden_dim, self.horizon * self.n_quantiles)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        forecast = self.forecast(h)
        if self.n_quantiles > 1:
            forecast = forecast.view(x.shape[0], self.horizon, self.n_quantiles)
        return self.backcast(h), forecast


class NBEATSForecaster(nn.Module):
    def __init__(
        self,
        n_features: int,
        lookback: int,
        horizon: int = 24,
        hidden_dim: int = 512,
        num_blocks: int = 4,
        dropout: float = 0.1,
        n_quantiles: int = 1,
    ) -> None:
        super().__init__()
        input_dim = int(n_features) * int(lookback)
        self.lookback = int(lookback)
        self.horizon = int(horizon)
        self.n_quantiles = max(1, int(n_quantiles))
        self.blocks = nn.ModuleList(
            [
                _NBEATSBlock(
                    input_dim=input_dim,
                    hidden_dim=int(hidden_dim),
                    horizon=int(horizon),
                    dropout=float(dropout),
                    n_quantiles=self.n_quantiles,
                )
                for _ in range(int(num_blocks))
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x.reshape(x.shape[0], -1)
        if self.n_quantiles > 1:
            forecast = torch.zeros(
                residual.shape[0],
                self.horizon,
                self.n_quantiles,
                dtype=residual.dtype,
                device=residual.device,
            )
        else:
            forecast = torch.zeros(
                residual.shape[0],
                self.horizon,
                dtype=residual.dtype,
                device=residual.device,
            )
        for block in self.blocks:
            backcast, block_forecast = block(residual)
            residual = residual - backcast
            forecast = forecast + block_forecast
        return forecast
