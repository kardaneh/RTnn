import torch
import torch.nn as nn
import sys
import time

class Attention(nn.Module):
    """
    Multi-head self-attention mechanism

    Args:
        embed_size (int): Total embedding size of the model.
        heads (int): Number of attention heads. Must divide embed_size evenly.
    """
    def __init__(self, embed_size, heads):
        super(Attention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=True)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=True)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=True)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        """
        Perform multi-head attention.

        Args:
            values (Tensor): Input tensor for values, shape (N, value_len, embed_size)
            keys (Tensor): Input tensor for keys, shape (N, key_len, embed_size)
            query (Tensor): Input tensor for queries, shape (N, query_len, embed_size)
            mask (Tensor, optional): Masking tensor to prevent attention to certain positions,
                                     shape (N, 1, 1, key_len)
        Returns:
            Tensor: Output after attention, shape (N, query_len, embed_size)
        """
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split embedding into multiple heads: (N, seq_len, heads, head_dim)
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        query = query.reshape(N, query_len, self.heads, self.head_dim)

        # Apply linear transformations
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(query)

        # Compute attention scores using dot product: (N, heads, query_len, key_len)
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        # Apply mask (if provided)
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        # Scale and apply softmax to get attention weights
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        # Multiply attention with values: (N, query_len, heads, head_dim)
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values])

        # Concatenate heads and pass through final linear layer
        out = out.reshape(N, query_len, self.heads * self.head_dim)

        out = self.fc_out(out)
        return out


class TransformerBlock(nn.Module):
    """
    A single Transformer block consisting of:

    1. Multi-head self-attention
    2. Add & LayerNorm
    3. Feed-forward network
    4. Add & LayerNorm

    Args:
        embed_size (int): Dimension of the input embeddings.
        heads (int): Number of attention heads.
        dropout (float): Dropout probability.
        forward_expansion (int): Factor to expand the embedding size in the feed-forward layer.

    Inputs:
        - value (Tensor): Value tensor of shape (N, seq_len, embed_size)
        - key (Tensor): Key tensor of shape (N, seq_len, embed_size)
        - query (Tensor): Query tensor of shape (N, seq_len, embed_size)
        - mask (Tensor, optional): Mask to prevent attention to certain positions,
                                   shape (N, 1, 1, seq_len) or broadcastable.

    Output:
        - out (Tensor): Tensor of shape (N, seq_len, embed_size)
    """
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()

        # Multi-head attention sublayer
        self.attention = Attention(embed_size, heads)

        # LayerNorm after attention and after feed-forward
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        # Position-wise feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        """
        Forward pass of the Transformer block.

        1. Applies multi-head attention with residual connection and layer norm.
        2. Applies feed-forward network with another residual connection and layer norm.

        Args:
            value, key, query (Tensor): Tensors with shape (N, seq_len, embed_size)
            mask (Tensor, optional): Attention mask, shape (N, 1, 1, seq_len)

        Returns:
            out (Tensor): Output of the block, shape (N, seq_len, embed_size)
        """

        # Self-attention
        attention = self.attention(value, key, query, mask)

        # Residual connection + LayerNorm + Dropout
        x = self.dropout(self.norm1(attention + query))
        
        # Feed-forward network
        forward = self.feed_forward(x)

        # Final residual connection + LayerNorm + Dropout
        out = self.dropout(self.norm2(forward + x))
        return out

class Encoder(nn.Module):
    """
    Transformer Encoder consisting of positional encoding,
    multiple Transformer blocks, and an output projection layer.

    Args:
        feature_channel (int): Number of input features per token.
        output_channel (int): Number of output channels for the final layer.
        embed_size (int): Embedding dimension used in the Transformer.
        num_layers (int): Number of Transformer blocks.
        heads (int): Number of attention heads.
        forward_expansion (int): Expansion factor in the feed-forward layer.
        seq_length (int): Maximum input sequence length.
        dropout (float): Dropout probability.
    """
    def __init__(
        self,
        feature_channel,
        output_channel,
        embed_size,
        num_layers,
        heads,
        forward_expansion,
        seq_length,
        dropout
    ):
        super(Encoder, self).__init__()

        self.embed_size = embed_size
        self.position_embedding = nn.Embedding(seq_length, embed_size)

        # Project raw input features to embed_size
        self.input_projection = nn.Linear(feature_channel, embed_size)
        self.input_activation = nn.ReLU()

        # Dropout after input embedding
        self.dropout = nn.Dropout(dropout)

        # Stack of Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(
                embed_size,
                heads,
                dropout=dropout,
                forward_expansion=forward_expansion,
            )
            for _ in range(num_layers)
        ])

        # Final output projection (Conv1d over time)
        self.final = nn.Conv1d(
            in_channels=embed_size,
            out_channels=output_channel,
            kernel_size=1,
            padding=0,
            bias=True
        )

    def forward(self, x, mask=None):
        """
        Args:
            x (Tensor): Input tensor of shape (batch_size, feature_channel, seq_len)
            mask (Tensor, optional): Attention mask (optional)

        Returns:
            Tensor: Output of shape (batch_size, output_channel, seq_len)
        """
        # Convert to shape (batch_size, seq_len, feature_channel)
        x = x.permute(0, 2, 1)
        N, seq_len, _ = x.shape

        # Positional encoding indices
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0).expand(N, seq_len)
        pos_embedding = self.position_embedding(positions)  # (N, seq_len, embed_size)

        # Project input and add positional encoding
        out = self.input_activation(self.input_projection(x))  # (N, seq_len, embed_size)
        out = self.dropout(out + pos_embedding)

        # Transformer layers
        for layer in self.layers:
            out = layer(out, out, out, mask)  # self-attention

        # Reshape to (N, embed_size, seq_len) for Conv1d
        out = out.permute(0, 2, 1)

        # Output projection
        out = self.final(out)  # (N, output_channel, seq_len)

        return out

def get_parameter_number(model):
    """
    Returns the total and trainable number of parameters in the model.
    """
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

def test_attention():
    print("Running test: Attention")

    net = Attention(embed_size=128, heads=2)
    values = torch.randn(32, 50, 128)
    keys   = torch.randn(32, 50, 128)
    query  = torch.randn(32, 50, 128)

    out = net(values, keys, query, mask=None)
    print(f"Output shape: {out.shape}")
    print(f"Params: {get_parameter_number(net)}\n")

def test_transformer_block():
    print("Running test: TransformerBlock")

    net = TransformerBlock(embed_size=128, heads=2, dropout=0.1, forward_expansion=2)
    values = torch.randn(32, 50, 128)
    keys   = torch.randn(32, 50, 128)
    query  = torch.randn(32, 50, 128)

    out = net(values, keys, query, mask=None)
    print(f"Output shape: {out.shape}")
    print(f"Params: {get_parameter_number(net)}\n")

def test_encoder():
    print("Running test: Encoder")

    model = Encoder(
        feature_channel=6,
        output_channel=4,
        embed_size=64,
        num_layers=2,
        heads=4,
        forward_expansion=4,
        seq_length=10,
        dropout=0.1
    )

    x = torch.randn(125760, 6, 10)  # (batch_size, feature_channel, seq_len)

    start = time.time()
    out = model(x)
    duration = time.time() - start

    print(f"Inference time: {duration:.4f}s")
    print(f"Output shape: {out.shape}")
    print(f"Params: {get_parameter_number(model)}\n")

def run_all_tests():
    print("="*30)
    test_attention()
    print("="*30)
    test_transformer_block()
    print("="*30)
    test_encoder()
    print("="*30)


if __name__ == "__main__":
    run_all_tests()
