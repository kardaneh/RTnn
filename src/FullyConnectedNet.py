import sys
import torch
import torch.nn as nn

# Add parent directory to path for module import
sys.path.append("..")
from DimChangeModule import DimChange  # Make sure this module exists


class FCBlock(nn.Module):
    """A fully connected block with Linear -> BatchNorm -> ReLU"""
    def __init__(self, in_features, out_features):
        super(FCBlock, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class FullyConnectedNet(nn.Module):
    """Fully connected neural network for 1D feature input"""
    def __init__(self, in_channels, out_channels,
                 num_hidden_layers, hidden_dim,
                 signal_length=55, dim_expand=0):
        super(FullyConnectedNet, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.signal_length = signal_length
        self.dim_expand = dim_expand

        # Input flattening layer
        self.input_layer = FCBlock(in_channels * signal_length, hidden_dim)

        # Intermediate hidden layers
        self.hidden_layers = nn.Sequential(*[
            FCBlock(hidden_dim, hidden_dim) for _ in range(num_hidden_layers)
        ])

        # Final projection layer
        self.output_layer = nn.Linear(hidden_dim, signal_length * out_channels)

        # Optional dimension change module
        self.dim_change = None
        if dim_expand > 0:
            self.dim_change = DimChange(signal_length, signal_length + dim_expand)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        x = self.input_layer(x)
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        x = x.view(batch_size, self.out_channels, -1)

        if self.dim_change:
            x = self.dim_change(x)
        return x


def test():
    model = FullyConnectedNet(
        in_channels=6,
        out_channels=4,
        num_hidden_layers=3,
        hidden_dim=32,
        signal_length=10,
        dim_expand=0
    )

    print("Model Summary:\n", model)

    dummy_input = torch.randn(105840, 6, 10)
    output = model(dummy_input)

    print(f"\nOutput shape: {output.shape}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total trainable parameters: {total_params}")


if __name__ == "__main__":
    test()
