# Copyright 2026 IPSL / CNRS / Sorbonne University
# Authors: Kazem Ardaneh
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

import torch
import torch.nn as nn


class FCBlock(nn.Module):
    """
    A fully connected block with linear layer, batch normalization, and ReLU activation.

    This module applies a linear transformation, followed by batch normalization,
    and then a ReLU activation function.

    Parameters
    ----------
    in_features : int
        Number of input features.
    out_features : int
        Number of output features.

    Attributes
    ----------
    linear : nn.Linear
        Linear transformation layer.
    bn : nn.BatchNorm1d
        Batch normalization layer.
    relu : nn.ReLU
        ReLU activation function.

    Examples
    --------
    >>> block = FCBlock(128, 64)
    >>> x = torch.randn(32, 128)
    >>> y = block(x)
    >>> y.shape
    torch.Size([32, 64])
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        """
        Initialize the FCBlock.

        Parameters
        ----------
        in_features : int
            Number of input features.
        out_features : int
            Number of output features.
        """
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the FCBlock.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, in_features).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, out_features).

        Notes
        -----
        The forward pass applies: ReLU(BatchNorm(Linear(x)))
        """
        return self.relu(self.bn(self.linear(x)))


class FCN(nn.Module):
    """
    Fully Connected Network with configurable depth and width.

    This model flattens the input sequence and processes it through a series
    of fully connected layers. It can optionally expand the sequence length
    using a linear transformation.

    Parameters
    ----------
    feature_channel : int
        Number of input features per time step.
    output_channel : int
        Number of output channels.
    num_layers : int
        Number of hidden layers.
    hidden_size : int
        Size of hidden layers.
    seq_length : int, optional
        Length of the input sequence. Default is 55.
    dim_expand : int, optional
        Number of time steps to expand the output sequence by.
        Default is 0 (no expansion).

    Attributes
    ----------
    feature_channel : int
        Number of input features.
    output_channel : int
        Number of output channels.
    seq_length : int
        Length of the input sequence.
    dim_expand : int
        Number of time steps to expand by.
    input_layer : FCBlock
        First fully connected layer.
    hidden_layers : nn.Sequential
        Stack of hidden layers.
    output_layer : nn.Linear
        Final output layer.
    dim_change : nn.Linear or None
        Optional layer for sequence length expansion.

    Examples
    --------
    >>> model = FCN(
    ...     feature_channel=6,
    ...     output_channel=4,
    ...     num_layers=3,
    ...     hidden_size=196,
    ...     seq_length=10
    ... )
    >>> x = torch.randn(32, 6, 10)
    >>> y = model(x)
    >>> y.shape
    torch.Size([32, 4, 10])
    """

    def __init__(
        self,
        feature_channel: int,
        output_channel: int,
        num_layers: int,
        hidden_size: int,
        seq_length: int = 55,
        dim_expand: int = 0,
    ) -> None:
        """
        Initialize the FCN model.

        Parameters
        ----------
        feature_channel : int
            Number of input features.
        output_channel : int
            Number of output channels.
        num_layers : int
            Number of hidden layers.
        hidden_size : int
            Size of hidden layers.
        seq_length : int, optional
            Length of the input sequence. Default is 55.
        dim_expand : int, optional
            Number of time steps to expand the output sequence by.
            Default is 0 (no expansion).

        Raises
        ------
        ValueError
            If num_layers is less than 1.
        """
        super().__init__()

        if num_layers < 1:
            raise ValueError(f"num_layers must be at least 1, got {num_layers}")

        self.feature_channel = feature_channel
        self.output_channel = output_channel
        self.seq_length = seq_length
        self.dim_expand = dim_expand

        # Input layer: flattens feature_channel * seq_length to hidden_size
        self.input_layer = FCBlock(feature_channel * seq_length, hidden_size)

        # Hidden layers: stack of FCBlocks
        self.hidden_layers = nn.Sequential(
            *[FCBlock(hidden_size, hidden_size) for _ in range(num_layers)]
        )

        # Output layer: projects hidden_size to output_channel * seq_length
        self.output_layer = nn.Linear(hidden_size, seq_length * output_channel)

        # Optional sequence length expansion
        self.dim_change = (
            nn.Linear(seq_length, seq_length + dim_expand) if dim_expand > 0 else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the FCN.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, feature_channel, seq_length).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, output_channel, seq_length + dim_expand)
            if dim_expand > 0, otherwise (batch_size, output_channel, seq_length).

        Notes
        -----
        The forward pass:
        1. Flattens the input to (batch_size, feature_channel * seq_length)
        2. Passes through FCBlocks
        3. Projects to output dimensions
        4. Reshapes to (batch_size, output_channel, seq_length)
        5. Optionally expands sequence length
        """
        batch_size = x.size(0)

        # Flatten input: (batch, features, seq) -> (batch, features * seq)
        x = x.view(batch_size, -1)

        # Forward through FC layers
        x = self.input_layer(x)
        x = self.hidden_layers(x)
        x = self.output_layer(x)

        # Reshape to output format: (batch, output_channel, seq_length)
        x = x.view(batch_size, self.output_channel, -1)

        # Optional sequence length expansion
        if self.dim_change:
            x = self.dim_change(x.transpose(1, 2)).transpose(1, 2)

        return x


class LayerPositionalEmbedding(nn.Module):
    def __init__(self, n_layers=10, embed_dim=16):
        super().__init__()
        self.embedding = nn.Embedding(n_layers, embed_dim)

    def forward(self, x):
        # x: (B, L, C)
        B, L, _ = x.shape
        idx = torch.arange(L, device=x.device)
        emb = self.embedding(idx)  # (L, E)
        emb = emb.unsqueeze(0).expand(B, -1, -1)  # (B, L, E)
        return torch.cat([x, emb], dim=-1)  # (B, L, C+E)


class VerticalRTColumnNet(nn.Module):
    """
    Two-stream RT emulator for vegetation canopies.

    Changes vs your original
    ------------------------
    1.  Coupled sweep: D and U interact at every layer via C_down/C_up,
        mirroring the γ2 cross-coupling in Eq.(2) of the paper.
        d_new = T_down * d  +  C_down * u  +  S_down
        u_new = T_up   * u  +  C_up   * d  +  S_up
        The upward sweep then refines U[l] using the same coupling,
        with D[l] already fixed.

    2.  Separate projection heads for D and U before merging.
        head_D(D[l]) + head_U(U[l]) + skip_proj(h[:,l])
        This preserves the physical identity of each stream.

    3.  Sigmoid on C_down / C_up keeps coupling coefficients in (0,1),
        consistent with γ2 being a positive scattering fraction.

    x shape in  : (B, C, L)      C = feature channels, L = 10 layers
    x shape out : (B, out_C, L)
    """

    def __init__(
        self,
        feature_channel=6,
        hidden=64,
        out_channel=4,
        n_layers=10,
        layer_embed_dim=16,
        dropout=0.1,
    ):
        super().__init__()
        self.out_channel = out_channel

        # --- Depth embedding ---
        self.layer_embed = LayerPositionalEmbedding(
            n_layers=n_layers,
            embed_dim=layer_embed_dim,
        )
        encoder_in = feature_channel + layer_embed_dim

        # --- Shared encoder ---
        self.encoder = nn.Sequential(
            nn.Linear(encoder_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # --- Downward stream ---
        # T_down: self-attenuation  ≈ exp(-γ1 Δτ),  constrained to (0,1)
        self.T_down = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_channel),
            nn.Sigmoid(),
        )
        # C_down: coupling from upward stream  ≈ γ2,  constrained to (0,1)
        self.C_down = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_channel),
            nn.Sigmoid(),
        )
        self.S_down = nn.Linear(hidden, out_channel)  # direct-beam source term

        # --- Surface boundary condition ---
        self.surface_bc = nn.Sequential(
            nn.Linear(out_channel, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, out_channel),
        )

        # --- Upward stream ---
        self.T_up = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_channel),
            nn.Sigmoid(),
        )
        # C_up: coupling from downward stream  ≈ γ2
        self.C_up = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_channel),
            nn.Sigmoid(),
        )
        self.S_up = nn.Linear(hidden, out_channel)

        # --- Residual skip ---
        self.skip_proj = nn.Linear(hidden, out_channel)

        # --- Separate output heads for each stream ---
        # Physical motivation: D[l] and U[l] represent different flux
        # directions; their contributions to albedo/transmittance differ.
        self.head_D = nn.Sequential(
            nn.Linear(out_channel, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, out_channel),
        )
        self.head_U = nn.Sequential(
            nn.Linear(out_channel, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, out_channel),
        )

    def forward(self, x):
        """
        x: (B, C, L)
        """
        # (B, C, L) → (B, L, C)
        x = x.permute(0, 2, 1)

        # Add depth embedding
        x = self.layer_embed(x)  # (B, L, C+E)

        # Encode all layers at once
        h = self.encoder(x)  # (B, L, H)
        B, L, _ = h.shape

        # ------------------------------------------------------------------
        # Downward sweep  (layer 0 = canopy top, L-1 = bottom)
        # Coupled: each step, D sees the current U estimate.
        # On the first pass U is zero (no upwelling above the canopy top).
        # ------------------------------------------------------------------
        D = []
        d = torch.zeros(B, self.out_channel, device=h.device)
        u = torch.zeros(B, self.out_channel, device=h.device)

        for nl in range(L):
            hl = h[:, nl]  # (B, H)
            Td = self.T_down(hl)  # (B, out_C) ∈ (0,1)
            Cd = self.C_down(hl)  # (B, out_C) ∈ (0,1)
            Sd = self.S_down(hl)  # (B, out_C)
            Tu = self.T_up(hl)
            Cu = self.C_up(hl)
            Su = self.S_up(hl)
            # Simultaneous update: each stream sees the other's prior state
            d_new = Td * d + Cd * u + Sd
            u_new = Tu * u + Cu * d + Su
            d, u = d_new, u_new
            D.append(d)

        # ------------------------------------------------------------------
        # Surface boundary condition
        # Physical meaning: bottom-of-canopy downward flux × surface
        # reflectance seeds the upward stream.
        # rs_surface_emu is already encoded in the features so surface_bc
        # implicitly learns the reflectance weighting.
        # ------------------------------------------------------------------
        u = self.surface_bc(D[-1])  # (B, out_C)

        # ------------------------------------------------------------------
        # Upward sweep  (L-1 → 0)
        # D[nl] is already fixed; upward stream is now refined with coupling.
        # ------------------------------------------------------------------
        U = [None] * L
        for nl in reversed(range(L)):
            hl = h[:, nl]
            Tu = self.T_up(hl)
            Cu = self.C_up(hl)
            Su = self.S_up(hl)
            # Couple back to the fixed downward flux at this layer
            u = Tu * u + Cu * D[nl] + Su
            U[nl] = u

        # ------------------------------------------------------------------
        # Per-layer output: separate heads + residual skip
        # ------------------------------------------------------------------
        out = []
        for nl in range(L):
            y = self.head_D(D[nl]) + self.head_U(U[nl])
            y = y + self.skip_proj(h[:, nl])  # residual from encoder
            out.append(y)

        return torch.stack(out, dim=2)  # (B, out_channel, L)
