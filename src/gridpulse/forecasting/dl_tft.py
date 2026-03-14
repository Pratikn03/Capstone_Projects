"""Lightweight transformer forecaster used as a TFT-style baseline."""
from __future__ import annotations

import torch
import torch.nn as nn


class TFTForecaster(nn.Module):
    def __init__(
        self,
        n_features: int,
        horizon: int = 24,
        d_model: int = 128,
        n_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        dim_feedforward: int = 256,
        n_quantiles: int = 1,
    ) -> None:
        super().__init__()
        self.horizon = int(horizon)
        self.n_quantiles = max(1, int(n_quantiles))
        self.input_proj = nn.Linear(n_features, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.attn_pool = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.query = nn.Parameter(torch.randn(1, 1, d_model))
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, self.horizon * self.n_quantiles),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_norm(self.input_proj(x))
        h = self.encoder(h)
        query = self.query.expand(h.shape[0], -1, -1)
        pooled, _ = self.attn_pool(query, h, h, need_weights=False)
        output = self.head(pooled.squeeze(1))
        if self.n_quantiles > 1:
            return output.view(output.shape[0], self.horizon, self.n_quantiles)
        return output
