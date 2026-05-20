# Copyright 2026 IPSL / CNRS / Sorbonne University
# Authors: Kazem Ardaneh
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

"""
Multi-layer perceptron architectures for structured and sequence-based modeling.

This module provides flexible and extensible implementations of multi-layer
perceptrons (MLPs) tailored for tasks such as radiative transfer emulation
and other scientific machine learning applications involving structured inputs.

The module includes:

- MLPBlock: A configurable fully connected block with optional normalization,
  activation, and dropout.
- MLP: A flexible MLP architecture supporting positional embeddings,
  residual connections, and customizable depth.
- MLPResidual: A residual MLP with skip connections across all hidden layers
  for improved gradient flow and training stability.

Features
--------
- Configurable hidden layer sizes and depth
- Support for multiple normalization strategies (batch norm, layer norm)
- Choice of activation functions (ReLU, GELU, SiLU)
- Optional dropout for regularization
- Residual connections for improved optimization
- Learnable positional embeddings for sequence-aware modeling
- Designed for flattened sequence inputs and structured data

Notes
-----
- Inputs are expected in the shape (batch_size, feature_channel, seq_length)
  and are internally flattened before processing.
- Positional embeddings, when enabled, are concatenated to the input features
  before passing through the network.
- Residual connections in ``MLP`` are applied globally, while ``MLPResidual``
  applies residual connections at every hidden layer.
- Layer normalization is applied to outputs for improved numerical stability.

Dependencies
------------
- torch
- torch.nn
- typing

Examples
--------
Basic MLP usage:

>>> model = MLP(
...     feature_channel=6,
...     output_channel=4,
...     seq_length=10,
...     hidden_sizes=[512, 256, 128]
... )
>>> x = torch.randn(32, 6, 10)
>>> y = model(x)

Using MLP with positional embeddings and residuals:

>>> model = MLP(
...     feature_channel=6,
...     output_channel=4,
...     seq_length=10,
...     use_positional_embedding=True,
...     use_residual=True
... )

Using MLPResidual:

>>> model = MLPResidual(
...     feature_channel=6,
...     output_channel=4,
...     seq_length=10,
...     hidden_size=256,
...     num_layers=4
... )
>>> x = torch.randn(16, 6, 10)
>>> y = model(x)
"""

import torch
import torch.nn as nn
from typing import List


class MLPBlock(nn.Module):
    """
    A single MLP block with linear layer, normalization, activation, and dropout.

    Parameters
    ----------
    in_features : int
        Number of input features.
    out_features : int
        Number of output features.
    dropout : float, optional
        Dropout rate. Default is 0.1.
    use_batch_norm : bool, optional
        Whether to use batch normalization. Default is True.
    use_layer_norm : bool, optional
        Whether to use layer normalization. Default is False.
    activation : str, optional
        Activation function ('relu', 'gelu', 'silu'). Default is 'relu'.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        dropout: float = 0.1,
        use_batch_norm: bool = True,
        use_layer_norm: bool = False,
        activation: str = "relu",
    ):
        super().__init__()

        self.linear = nn.Linear(in_features, out_features)

        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm

        if use_batch_norm:
            self.bn = nn.BatchNorm1d(out_features)
        if use_layer_norm:
            self.ln = nn.LayerNorm(out_features)

        if activation == "relu":
            self.act = nn.ReLU()
        elif activation == "gelu":
            self.act = nn.GELU()
        elif activation == "silu":
            self.act = nn.SiLU()
        else:
            self.act = nn.ReLU()

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.linear(x)

        if self.use_batch_norm:
            x = self.bn(x)
        if self.use_layer_norm:
            x = self.ln(x)

        x = self.act(x)
        x = self.dropout(x)

        return x


class MLP(nn.Module):
    """
    Multi-Layer Perceptron for radiative transfer emulation.

    Parameters
    ----------
    feature_channel : int
        Number of input features per time step.
    output_channel : int
        Number of output channels.
    seq_length : int
        Length of the input sequence.
    hidden_sizes : List[int], optional
        List of hidden layer sizes. Default is [512, 256, 128].
    dropout : float, optional
        Dropout rate. Default is 0.1.
    use_batch_norm : bool, optional
        Whether to use batch normalization. Default is True.
    use_layer_norm : bool, optional
        Whether to use layer normalization. Default is False.
    use_residual : bool, optional
        Whether to use residual connections. Default is False.
    activation : str, optional
        Activation function ('relu', 'gelu', 'silu'). Default is 'relu'.
    use_positional_embedding : bool, optional
        Whether to add positional embeddings. Default is True.
    positional_embed_dim : int, optional
        Dimension of positional embeddings. Default is 16.
    """

    def __init__(
        self,
        feature_channel: int,
        output_channel: int,
        seq_length: int = 10,
        hidden_sizes: List[int] = None,
        dropout: float = 0.1,
        use_batch_norm: bool = True,
        use_layer_norm: bool = False,
        use_residual: bool = False,
        activation: str = "relu",
        use_positional_embedding: bool = True,
        positional_embed_dim: int = 16,
    ):
        super().__init__()

        if hidden_sizes is None:
            hidden_sizes = [512, 256, 128]

        self.feature_channel = feature_channel
        self.output_channel = output_channel
        self.seq_length = seq_length
        self.hidden_sizes = hidden_sizes
        self.use_residual = use_residual
        self.use_positional_embedding = use_positional_embedding

        input_size = feature_channel * seq_length

        if use_positional_embedding:
            self.positional_embed = nn.Embedding(seq_length, positional_embed_dim)
            input_size += positional_embed_dim * seq_length
        else:
            self.positional_embed = None

        self.layers = nn.ModuleList()
        prev_size = input_size

        for hidden_size in hidden_sizes:
            self.layers.append(
                MLPBlock(
                    prev_size,
                    hidden_size,
                    dropout=dropout,
                    use_batch_norm=use_batch_norm,
                    use_layer_norm=use_layer_norm,
                    activation=activation,
                )
            )
            prev_size = hidden_size

        output_size = output_channel * seq_length
        self.output_layer = nn.Linear(prev_size, output_size)
        self.output_dropout = nn.Dropout(dropout)
        self.final_norm = nn.LayerNorm(output_size)

        if use_residual and input_size != hidden_sizes[-1]:
            self.residual_proj = nn.Linear(input_size, hidden_sizes[-1])
        else:
            self.residual_proj = None

    def _add_positional_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional embeddings to the flattened input."""
        batch_size = x.shape[0]
        x_reshaped = x.reshape(batch_size, self.seq_length, -1)
        positions = torch.arange(self.seq_length, device=x.device)
        pos_embed = self.positional_embed(positions)
        pos_embed = pos_embed.unsqueeze(0).expand(batch_size, -1, -1)
        x_combined = torch.cat([x_reshaped, pos_embed], dim=-1)
        return x_combined.reshape(batch_size, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)

        if self.use_positional_embedding and self.positional_embed is not None:
            x = self._add_positional_embedding(x)

        residual = x if self.use_residual else None
        if self.residual_proj is not None:
            residual = self.residual_proj(residual)

        for layer in self.layers:
            x = layer(x)

        if self.use_residual and residual is not None:
            x = x + residual

        x = self.output_layer(x)
        x = self.output_dropout(x)
        x = self.final_norm(x)
        x = x.reshape(batch_size, self.output_channel, self.seq_length)

        return x


class MLPResidual(nn.Module):
    """
    MLP with residual connections between all layers.

    Parameters
    ----------
    feature_channel : int
        Number of input features.
    output_channel : int
        Number of output channels.
    seq_length : int
        Sequence length.
    hidden_size : int
        Size of hidden layers.
    num_layers : int
        Number of hidden layers.
    dropout : float, optional
        Dropout rate. Default is 0.1.
    """

    def __init__(
        self,
        feature_channel: int,
        output_channel: int,
        seq_length: int = 10,
        hidden_size: int = 256,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.feature_channel = feature_channel
        self.output_channel = output_channel
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        input_size = feature_channel * seq_length
        output_size = output_channel * seq_length

        self.input_proj = nn.Linear(input_size, hidden_size)
        self.input_norm = nn.LayerNorm(hidden_size)
        self.input_act = nn.ReLU()
        self.input_dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList()
        for _ in range(num_layers):
            block = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.blocks.append(block)

        self.output_proj = nn.Linear(hidden_size, output_size)
        self.output_norm = nn.LayerNorm(output_size)
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connections."""
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)

        x = self.input_proj(x)
        x = self.input_norm(x)
        x = self.input_act(x)
        x = self.input_dropout(x)

        for block in self.blocks:
            x = x + block(x)

        x = self.output_proj(x)
        x = self.output_norm(x)
        x = self.output_dropout(x)
        x = x.reshape(batch_size, self.output_channel, self.seq_length)

        return x
