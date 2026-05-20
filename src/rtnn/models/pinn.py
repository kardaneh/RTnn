import torch
import torch.nn as nn

"""
Physics-inspired neural network architectures for vertical profile modeling.
These models are designed to capture the structured interactions in radiative transfer processes, particularly in vegetation canopies. The key innovation is the
two-stream formulation that mimics the coupled upward and downward fluxes in radiative transfer, with learnable coupling coefficients and separate output heads for each stream.
"""


class LayerPositionalEmbedding(nn.Module):
    """
    Learnable positional embedding for layer-wise data.
    This module provides a learnable embedding for each layer in a vertical profile, allowing the model to distinguish between different physical layers (e.g., canopy levels). The embedding is added to the input features before processing.
    """

    def __init__(self, n_layers=10, embed_dim=16):
        super().__init__()
        self.embedding = nn.Embedding(n_layers, embed_dim)

    def forward(self, x):
        """
        Forward pass for the positional embedding.
        """
        # x: (B, L, C)
        B, L, _ = x.shape
        idx = torch.arange(L, device=x.device)
        emb = self.embedding(idx)  # (L, E)
        emb = emb.unsqueeze(0).expand(B, -1, -1)  # (B, L, E)
        return torch.cat([x, emb], dim=-1)  # (B, L, C+E)


class PINN(nn.Module):
    """
    Two-stream RT emulator for vegetation canopies.

    This architecture is inspired by the coupled radiative transfer equations:

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
