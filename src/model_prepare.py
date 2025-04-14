from ConvLSTM1D import RNN_LSTM, RNN_GRU


def load_model(model_name, device, feature_channel, signal_length):
    """
    Load and initialize a recurrent neural network model based on the model name.

    Args:
        model_name (str): Name of the model to load. Supported values include:
                          "LSTM", "LSTM_32_5", "LSTM_32_3", "LSTM_16_1", "GRU".
        device (torch.device): The device on which to place the model.
        feature_channel (int): Number of input features per time step.
        signal_length (int): Length of the signal (not directly used in this function).

    Returns:
        nn.Module: An initialized PyTorch model ready to be used.

    Raises:
        ValueError: If the given model name is not implemented.
    """
    model_configs = {
        "LSTM": {"class": RNN_LSTM, "hidden_size": 96, "num_layers": 5},
        "LSTM_32_5": {"class": RNN_LSTM, "hidden_size": 32, "num_layers": 5},
        "LSTM_32_3": {"class": RNN_LSTM, "hidden_size": 32, "num_layers": 3},
        "LSTM_16_1": {"class": RNN_LSTM, "hidden_size": 16, "num_layers": 1},
        "GRU": {"class": RNN_GRU, "hidden_size": 128, "num_layers": 5},
    }

    if model_name not in model_configs:
        raise ValueError(f"Model '{model_name}' is not implemented.")

    config = model_configs[model_name]
    model = config["class"](
        feature_channel=feature_channel,
        output_channel=4,
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
    )

    return model
