"""Enhanced Temporal Convolutional Network (TCN) forecaster with skip connections."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

class _Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        """Trim padding on the right to keep causal convolutions."""
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size]


class _TemporalBlock(nn.Module):
    def __init__(
        self, 
        in_ch: int, 
        out_ch: int, 
        kernel_size: int, 
        dilation: int, 
        dropout: float,
        use_layer_norm: bool = True,
    ):
        """Enhanced TCN block with optional layer normalization."""
        super().__init__()
        padding = (kernel_size - 1) * dilation
        
        self.conv1 = weight_norm(nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation))
        self.chomp1 = _Chomp1d(padding)
        self.relu1 = nn.GELU()  # GELU activation for smoother gradients
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, dilation=dilation))
        self.chomp2 = _Chomp1d(padding)
        self.relu2 = nn.GELU()
        self.drop2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.drop1, 
            self.conv2, self.chomp2, self.relu2, self.drop2
        )
        
        # Layer normalization for stability
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.norm = nn.LayerNorm(out_ch)
        
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.final_relu = nn.GELU()

    def forward(self, x):
        """Forward pass with residual connection and optional layer norm."""
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        out = out + res  # Residual connection
        
        if self.use_layer_norm:
            # Layer norm expects (B, T, C), so transpose
            out = out.transpose(1, 2)
            out = self.norm(out)
            out = out.transpose(1, 2)
            
        return self.final_relu(out)


class _TemporalConvNet(nn.Module):
    def __init__(
        self, 
        num_inputs: int, 
        num_channels: list[int], 
        kernel_size: int = 3, 
        dropout: float = 0.1,
        dilation_base: int = 2,
        use_skip_connections: bool = False,
    ):
        """Stack multiple temporal blocks with increasing dilation.
        
        Args:
            num_inputs: Number of input channels
            num_channels: List of output channels for each block
            kernel_size: Convolution kernel size
            dropout: Dropout rate
            dilation_base: Base for exponential dilation (default 2)
            use_skip_connections: Add skip connections from all layers to output
        """
        super().__init__()
        self.use_skip_connections = use_skip_connections
        
        layers = []
        skip_dims = []
        for i, out_ch in enumerate(num_channels):
            dilation = dilation_base ** i
            in_ch = num_inputs if i == 0 else num_channels[i - 1]
            layers.append(_TemporalBlock(in_ch, out_ch, kernel_size, dilation, dropout))
            if use_skip_connections:
                skip_dims.append(out_ch)
                
        self.layers = nn.ModuleList(layers)
        
        # Skip connection projection to final dimension
        if use_skip_connections:
            self.skip_proj = nn.Conv1d(sum(skip_dims), num_channels[-1], 1)
            
    def forward(self, x):
        """Forward pass through the stacked temporal blocks."""
        if self.use_skip_connections:
            skip_outputs = []
            for layer in self.layers:
                x = layer(x)
                skip_outputs.append(x)
            # Concatenate all skip connections and project
            skip_cat = torch.cat(skip_outputs, dim=1)
            return self.skip_proj(skip_cat)
        else:
            for layer in self.layers:
                x = layer(x)
            return x


class TCNForecaster(nn.Module):
    def __init__(
        self, 
        n_features: int, 
        num_channels: list[int], 
        kernel_size: int = 3, 
        dropout: float = 0.1,
        dilation_base: int = 2,
        use_skip_connections: bool = False,
        horizon: int = 24,
        n_quantiles: int = 1,
    ):
        """Enhanced TCN with skip connections and configurable dilation.
        
        Args:
            n_features: Number of input features
            num_channels: List of channel sizes for each TCN block
            kernel_size: Convolution kernel size
            dropout: Dropout rate
            dilation_base: Base for exponential dilation growth
            use_skip_connections: Aggregate features from all layers
            horizon: Forecast horizon (number of future steps)
        """
        super().__init__()
        if not num_channels:
            raise ValueError("num_channels must be a non-empty list")
            
        self.horizon = horizon
        self.n_quantiles = max(1, int(n_quantiles))
        
        # Input projection
        self.input_proj = nn.Conv1d(n_features, n_features, 1)
        self.input_norm = nn.LayerNorm(n_features)
        
        # TCN backbone
        self.tcn = _TemporalConvNet(
            n_features, 
            num_channels, 
            kernel_size=kernel_size, 
            dropout=dropout,
            dilation_base=dilation_base,
            use_skip_connections=use_skip_connections,
        )
        
        # Output head
        self.head = nn.Sequential(
            nn.Conv1d(num_channels[-1], num_channels[-1], 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(num_channels[-1], self.n_quantiles, 1),
        )

    def forward(self, x):
        """Forward pass: (batch, time, features) -> (batch, time) or (batch, horizon)."""
        # x: (B, T, F) -> (B, F, T)
        x = x.transpose(1, 2)
        
        # Input projection with layer norm
        x = self.input_proj(x)
        x = x.transpose(1, 2)
        x = self.input_norm(x)
        x = x.transpose(1, 2)
        
        # TCN encoding
        y = self.tcn(x)
        y = self.head(y)
        if self.n_quantiles > 1:
            return y.transpose(1, 2)  # (B, T, Q)
        return y.squeeze(1)  # (B, T)
