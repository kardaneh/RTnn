import torch
import torch.nn as nn
from DimChangeModule import DimChange

class FCBlock(nn.Module):
    """A fully connected block with linear, batch norm and ReLU activation"""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.linear(x)))


class FCN(nn.Module):
    """Fully Connected Network with configurable depth and width"""
    def __init__(
        self,
        feature_channel,
        output_channel,
        num_layers,
        hidden_size,
        seq_length = 55,
        dim_expand = 0,
    ):
        super().__init__()
        
        self.feature_channel = feature_channel
        self.output_channel = output_channel
        self.seq_length = seq_length
        self.dim_expand = dim_expand

        self.input_layer = FCBlock(feature_channel * seq_length, hidden_size)
        
        self.hidden_layers = nn.Sequential(
            *[FCBlock(hidden_size, hidden_size) for _ in range(num_layers)]
        )
        
        self.output_layer = nn.Linear(hidden_size, seq_length * output_channel)
        
        self.dim_change = nn.Linear(seq_length, seq_length + dim_expand) if dim_expand > 0 else None

    def forward(self, x):
        """Forward pass"""
        batch_size = x.size(0)
        
        x = x.view(batch_size, -1)
        
        x = self.input_layer(x)
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        
        x = x.view(batch_size, self.output_channel, -1)
        
        if self.dim_change:
            x = self.dim_change(x.transpose(1, 2)).transpose(1, 2)
            
        return x

def test():
    model = FCN(
        feature_channel=6,
        output_channel=4,
        num_layers=3,
        hidden_size=196,
        seq_length=10,
        dim_expand=0,
    )

    print("Model Summary:\n", model)

    dummy_input = torch.randn(16*7860, 6, 10)
    output = model(dummy_input)

    print(f"\nOutput shape: {output.shape}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total trainable parameters: {total_params}")


if __name__ == "__main__":
    test()
