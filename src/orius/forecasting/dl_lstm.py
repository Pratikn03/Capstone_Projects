"""Forecasting: Enhanced LSTM sequence model with attention and bidirectional support."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _MultiHeadAttention(nn.Module):
    """Multi-head self-attention for sequence modeling."""

    def __init__(self, hidden_size: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})")

        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Apply self-attention: (batch, seq, hidden) -> (batch, seq, hidden)."""
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class LSTMForecaster(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        horizon: int = 48,
        bidirectional: bool = False,
        attention: bool = False,
        residual: bool = False,
        n_quantiles: int = 1,
    ):
        """Enhanced LSTM with optional bidirectional, attention, and residual connections.

        Args:
            n_features: Number of input features
            hidden_size: LSTM hidden state size
            num_layers: Number of stacked LSTM layers
            dropout: Dropout rate
            horizon: Forecast horizon (number of future steps to predict)
            bidirectional: Use bidirectional LSTM (doubles effective hidden size)
            attention: Add multi-head attention after LSTM
            residual: Use residual connections from input embedding
        """
        super().__init__()
        self.bidirectional = bidirectional
        self.attention = attention
        self.residual = residual
        self.hidden_size = hidden_size
        self.horizon = int(horizon)
        self.n_quantiles = max(1, int(n_quantiles))

        # Input projection with layer norm
        self.input_proj = nn.Linear(n_features, hidden_size)
        self.input_norm = nn.LayerNorm(hidden_size)

        # Recurrent backbone
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )

        # Effective hidden size after LSTM
        lstm_out_size = hidden_size * 2 if bidirectional else hidden_size

        # Post-LSTM layer norm
        self.lstm_norm = nn.LayerNorm(lstm_out_size)

        # Optional attention mechanism
        if attention:
            self.attn = _MultiHeadAttention(lstm_out_size, num_heads=4, dropout=dropout)
            self.attn_norm = nn.LayerNorm(lstm_out_size)

        # Optional residual projection (match dimensions)
        if residual and lstm_out_size != hidden_size:
            self.res_proj = nn.Linear(hidden_size, lstm_out_size)
        else:
            self.res_proj = None

        # Output head: predict all horizon steps
        self.head = nn.Sequential(
            nn.Linear(lstm_out_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, self.horizon * self.n_quantiles),
        )

    def forward(self, x):
        """Forward pass: (batch, time, features) -> (batch, horizon)."""
        # Input projection
        h = self.input_proj(x)
        h = self.input_norm(h)
        res = h  # For residual connection

        # LSTM encoding
        lstm_out, _ = self.lstm(h)
        lstm_out = self.lstm_norm(lstm_out)

        # Optional residual from input embedding
        if self.residual:
            if self.res_proj is not None:
                res = self.res_proj(res)
            lstm_out = lstm_out + res

        # Optional attention
        if self.attention:
            attn_out = self.attn(lstm_out)
            lstm_out = self.attn_norm(lstm_out + attn_out)

        # Take last timestep and project to horizon
        last_step = lstm_out[:, -1, :]
        y = self.head(last_step)
        if self.n_quantiles > 1:
            return y.view(y.shape[0], self.horizon, self.n_quantiles)
        return y
