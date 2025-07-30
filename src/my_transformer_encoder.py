# coding: utf-8
import torch
import torch.nn as nn


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, seq_length, embed_size):
        super().__init__()
        self.pos_embedding = nn.Embedding(seq_length, embed_size)

    def forward(self, x):
        N, seq_len, _ = x.size()
        positions = (
            torch.arange(0, seq_len).unsqueeze(0).expand(N, seq_len).to(x.device)
        )
        return x + self.pos_embedding(positions)


class MyTransformer(nn.Module):
    def __init__(
        self,
        feature_channel=6,
        output_channel=4,
        embed_size=32,
        seq_length=10,
        num_layers=3,
        nhead=1,
        forward_expansion=1,
        dropout=0.0,
    ):
        super().__init__()
        self.input_proj = nn.Linear(feature_channel, embed_size)
        self.pos_encoding = LearnedPositionalEncoding(seq_length, embed_size)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_size,
            nhead=nhead,
            dim_feedforward=embed_size * forward_expansion,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(embed_size, output_channel)

    def forward(self, x):
        # x shape: (batch, feature_channel, seq_len)
        x = x.permute(0, 2, 1)  # (batch, seq_len, feature_channel)
        x = self.input_proj(x)  # (batch, seq_len, embed_size)
        x = self.pos_encoding(x)
        out = self.encoder(x)  # (batch, seq_len, embed_size)
        out = self.output_proj(out)  # (batch, seq_len, output_channel)
        return out.permute(0, 2, 1)  # (batch, output_channel, seq_len)


def test():
    model = MyTransformer()
    x = torch.randn(105840, 6, 10)
    y = model(x)
    print("Output shape:", y.shape)


if __name__ == "__main__":
    test()
