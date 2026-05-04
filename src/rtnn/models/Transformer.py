# Copyright 2026 IPSL / CNRS / Sorbonne University
# Authors: Kazem Ardaneh
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

import torch
import torch.nn as nn
from typing import Optional


class EncoderTorch(nn.Module):
    def __init__(
        self,
        feature_channel: int,
        output_channel: int,
        embed_size: int,
        num_layers: int,
        heads: int,
        forward_expansion: int,
        seq_length: int,
        dropout: float,
    ) -> None:
        super().__init__()

        if num_layers < 1:
            raise ValueError(f"num_layers must be at least 1, got {num_layers}")

        self.embed_size = embed_size
        self.seq_length = seq_length

        # Input projection
        self.input_proj = nn.Linear(feature_channel, embed_size)

        # Positional embedding
        self.position_embedding = nn.Embedding(seq_length, embed_size)

        # PyTorch TransformerEncoderLayer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_size,
            nhead=heads,
            dim_feedforward=forward_expansion * embed_size,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=False,
        )

        # Stack layers
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.dropout = nn.Dropout(dropout)

        # Final projection
        self.final = nn.Conv1d(embed_size, output_channel, kernel_size=1)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x: (batch, feature_channel, seq_length)
        """

        # (batch, seq, feature)
        x = x.permute(0, 2, 1)
        N, seq_len, _ = x.shape

        # Positional encoding
        positions = (
            torch.arange(0, seq_len, device=x.device).unsqueeze(0).expand(N, seq_len)
        )
        pos_embed = self.position_embedding(positions)

        # Input projection + position
        x = self.input_proj(x)
        x = x + pos_embed
        x = self.dropout(x)

        # Transformer encoder
        x = self.encoder(x, mask=mask, src_key_padding_mask=src_key_padding_mask)

        # Back to (batch, channels, seq)
        x = x.permute(0, 2, 1)
        x = self.final(x)

        return x
