"""Temporal Convolutional Network (TCN) forecaster."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

class _Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        # Key: prepare features/targets and train or evaluate models
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size]

class _TemporalBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = weight_norm(nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation))
        self.chomp1 = _Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, dilation=dilation))
        self.chomp2 = _Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.drop1, self.conv2, self.chomp2, self.relu2, self.drop2)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.final_relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.final_relu(out + res)

class _TemporalConvNet(nn.Module):
    def __init__(self, num_inputs: int, num_channels: list[int], kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        layers = []
        for i, out_ch in enumerate(num_channels):
            dilation = 2 ** i
            in_ch = num_inputs if i == 0 else num_channels[i - 1]
            layers.append(_TemporalBlock(in_ch, out_ch, kernel_size, dilation, dropout))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class TCNForecaster(nn.Module):
    def __init__(self, n_features: int, num_channels: list[int], kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        if not num_channels:
            raise ValueError("num_channels must be a non-empty list")
        self.tcn = _TemporalConvNet(n_features, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.head = nn.Conv1d(num_channels[-1], 1, kernel_size=1)

    def forward(self, x):
        # x: (B, T, F) -> (B, F, T)
        x = x.transpose(1, 2)
        y = self.tcn(x)
        y = self.head(y)  # (B, 1, T)
        return y.squeeze(1)  # (B, T)
