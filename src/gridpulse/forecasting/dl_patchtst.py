"""PatchTST-style transformer forecaster for multivariate hourly sequences."""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class PatchTSTForecaster(nn.Module):
    def __init__(
        self,
        n_features: int,
        lookback: int,
        horizon: int = 24,
        patch_len: int = 24,
        stride: int = 12,
        d_model: int = 128,
        n_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        dim_feedforward: int = 256,
        n_quantiles: int = 1,
    ) -> None:
        super().__init__()
        self.lookback = int(lookback)
        self.horizon = int(horizon)
        self.n_quantiles = max(1, int(n_quantiles))
        self.patch_len = max(1, int(patch_len))
        self.stride = max(1, int(stride))
        n_patches = max(1, math.floor((self.lookback - self.patch_len) / self.stride) + 1)

        self.patch_proj = nn.Linear(self.patch_len * int(n_features), d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches, d_model) * 0.02)
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
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, self.horizon * self.n_quantiles),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] < self.patch_len:
            pad_steps = self.patch_len - x.shape[1]
            x = torch.nn.functional.pad(x, (0, 0, pad_steps, 0))
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        # unfold yields (B, n_patches, patch_len, F)
        patches = patches.contiguous().reshape(x.shape[0], patches.shape[1], -1)
        h = self.patch_proj(patches)
        pos = self.pos_embed[:, : h.shape[1], :]
        h = self.encoder(h + pos)
        pooled = h.mean(dim=1)
        output = self.head(pooled)
        if self.n_quantiles > 1:
            return output.view(output.shape[0], self.horizon, self.n_quantiles)
        return output
