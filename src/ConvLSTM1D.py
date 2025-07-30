import torch
import torch.nn as nn


class RNN_LSTM(nn.Module):
    """
    LSTM-based RNN module with a final 1D convolutional projection layer.

    Args:
        feature_channel (int): Number of input features per time step.
        output_channel (int): Number of output channels after projection.
        hidden_size (int): Hidden state size of the LSTM.
        num_layers (int): Number of stacked LSTM layers.
    """

    def __init__(self, feature_channel, output_channel, hidden_size, num_layers):
        super(RNN_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_channel = output_channel

        self.lstm = nn.LSTM(
            input_size=feature_channel,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

        self.final = nn.Conv1d(
            in_channels=2 * hidden_size,
            out_channels=output_channel,
            kernel_size=1,
            padding=0,
            bias=True,
        )

    def forward(self, x):
        """
        Forward pass of the LSTM-based model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, feature_channel, sequence_length).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_channel, sequence_length).
        """
        x = torch.permute(x, [0, 2, 1])
        h0 = torch.zeros(
            2 * self.num_layers, x.shape[0], self.hidden_size, requires_grad=False
        ).to(x.device)
        c0 = torch.zeros(
            2 * self.num_layers, x.shape[0], self.hidden_size, requires_grad=False
        ).to(x.device)

        hidden = (h0, c0)
        (h0, c0) = hidden

        out, _ = self.lstm(x, (h0, c0))
        out = torch.permute(out, [0, 2, 1])
        out = self.final(out)
        return out


class RNN_GRU(nn.Module):
    """
    GRU-based RNN module with a final 1D convolutional projection layer.

    Args:
        feature_channel (int): Number of input features per time step.
        output_channel (int): Number of output channels after projection.
        hidden_size (int): Hidden state size of the GRU.
        num_layers (int): Number of stacked GRU layers.
    """

    def __init__(self, feature_channel, output_channel, hidden_size, num_layers):
        super(RNN_GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_channel = output_channel

        self.lstm = nn.GRU(
            input_size=feature_channel,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

        self.final = nn.Conv1d(
            in_channels=2 * hidden_size,
            out_channels=output_channel,
            kernel_size=1,
            padding=0,
            bias=True,
        )

    def forward(self, x):
        """
        Forward pass of the GRU-based model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, feature_channel, sequence_length).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_channel, sequence_length).
        """
        x = torch.permute(x, [0, 2, 1])
        h0 = torch.zeros(
            2 * self.num_layers, x.shape[0], self.hidden_size, requires_grad=False
        ).to(x.device)

        out, _ = self.lstm(x, h0)
        out = torch.permute(out, [0, 2, 1])
        out = self.final(out)
        return out


def test(model_type="lstm"):
    """
    Test function for the RNN_LSTM and RNN_GRU classes.
    Creates a model, runs a forward pass, and prints output shape and parameter count.

    Args:
        model_type (str): Choose between "lstm" or "gru" model to test.
    """
    feature_channel = 6
    output_channel = 4
    hidden_size = 32
    num_layers = 3
    batch_size = 16 * 7860
    sequence_length = 10

    if model_type == "lstm":
        net = RNN_LSTM(feature_channel, output_channel, hidden_size, num_layers)
    elif model_type == "gru":
        net = RNN_GRU(feature_channel, output_channel, hidden_size, num_layers)
    else:
        raise ValueError("Invalid model type. Choose 'lstm' or 'gru'.")

    x = torch.randn(batch_size, feature_channel, sequence_length)
    y = net(x)

    print(f"Output shape: {y.size()}")
    total_params = sum(p.numel() for p in net.parameters())
    print(f"Total parameters: {total_params}")

    print("RNN_LSTM(")
    for name, param in net.named_parameters():
        shape_str = f"{list(param.shape)}"

        # Parameter type
        if "weight_ih" in name:
            param_type = "W_ih (input → hidden)"
        elif "weight_hh" in name:
            param_type = "W_hh (hidden → hidden)"
        elif "bias_ih" in name:
            param_type = "b_ih"
        elif "bias_hh" in name:
            param_type = "b_hh"
        elif "final.weight" in name:
            param_type = "Final Linear Weights"
        elif "final.bias" in name:
            param_type = "Final Linear Bias"
        else:
            param_type = "Other"

        # Layer and direction
        layer_info = ""
        if "_l0" in name:
            layer_info = " [L0]"
        elif "_l1" in name:
            layer_info = " [L1]"
        elif "_l2" in name:
            layer_info = " [L2]"

        direction = (
            " (→)"
            if "reverse" not in name and "lstm" in name
            else " (←)"
            if "reverse" in name
            else ""
        )

        print(f"  {name:<30} {shape_str:<20} {param_type}{layer_info}{direction}")
    print(")")


if __name__ == "__main__":
    test("lstm")  # Change to "gru" to test RNN_GRU
