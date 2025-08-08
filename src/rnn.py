import torch
import torch.nn as nn


class BaseRNN(nn.Module):
    """Base class for RNN modules"""
    def __init__(self, feature_channel, output_channel, hidden_size, num_layers, rnn_type):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_channel = output_channel
        
        rnn_class = nn.LSTM if rnn_type == 'lstm' else nn.GRU
        self.rnn = rnn_class(
            input_size=feature_channel,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        
        self.final = nn.Conv1d(
            in_channels=2 * hidden_size,
            out_channels=output_channel,
            kernel_size=1,
            padding=0,
            bias=True
        )

    def init_hidden(self, batch_size, device):
        """Initialize hidden state"""
        return torch.zeros(
            2 * self.num_layers,
            batch_size,
            self.hidden_size,
            device=device,
            requires_grad=False
        )

    def forward(self, x):
        """Forward pass for RNN models"""
        x = x.permute(0, 2, 1)
        
        hidden = self.init_hidden(x.size(0), x.device)
        if isinstance(self.rnn, nn.LSTM):
            hidden = (hidden, hidden.clone())
        
        out, _ = self.rnn(x, hidden)
        
        return self.final(out.permute(0, 2, 1))


class RNN_LSTM(BaseRNN):
    """LSTM-based RNN module"""
    def __init__(self, feature_channel, output_channel, hidden_size, num_layers):
        super().__init__(feature_channel, output_channel, hidden_size, num_layers, 'lstm')


class RNN_GRU(BaseRNN):
    """GRU-based RNN module"""
    def __init__(self, feature_channel, output_channel, hidden_size, num_layers):
        super().__init__(feature_channel, output_channel, hidden_size, num_layers, 'gru')


def test_model(model_type="lstm"):
    """Test function for RNN models"""
    config = {
        'feature_channel': 6,
        'output_channel': 4,
        'hidden_size': 64,
        'num_layers': 2,
        'batch_size': 16 * 7860,
        'sequence_length': 10
    }
    
    model_class = RNN_LSTM if model_type == "lstm" else RNN_GRU
    net = model_class(
        feature_channel=config['feature_channel'],
        output_channel=config['output_channel'],
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers']
    )
    
    x = torch.randn(config['batch_size'], config['feature_channel'], config['sequence_length'])
    y = net(x)
    
    print(f"Output shape: {y.size()}")
    print(f"Total parameters: {sum(p.numel() for p in net.parameters())}")
    
    print(f"\n{model_class.__name__} architecture:")
    for name, param in net.named_parameters():
        param_type = "Weight" if "weight" in name else "Bias"
        direction = "(forward)" if "reverse" not in name else "(backward)"
        layer_num = name.split('_')[1][1:] if '_l' in name else ""
        
        print(f"  {name:<30} {str(list(param.shape)):<20} {param_type} {direction} {'Layer '+layer_num if layer_num else ''}")


if __name__ == "__main__":
    test_model("gru")
