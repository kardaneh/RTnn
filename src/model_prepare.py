from ConvLSTM1D import RNN_LSTM, RNN_GRU
from my_transformer_encoder import MyTransformer
from Transformer import Encoder
from FullyConnectedNet import FullyConnectedNet
from UNet1D import UNET


def load_model(model_name, device, feature_channel, signal_length):
    """ """
    model_configs = {
        "LSTM": {"class": RNN_LSTM, "hidden_size": 96, "num_layers": 5},
        "LSTM_32_5": {"class": RNN_LSTM, "hidden_size": 32, "num_layers": 5},
        "LSTM_32_3": {"class": RNN_LSTM, "hidden_size": 32, "num_layers": 3},
        "LSTM_96_3": {"class": RNN_LSTM, "hidden_size": 96, "num_layers": 3},
        "LSTM_64_2": {"class": RNN_LSTM, "hidden_size": 64, "num_layers": 2},
        "GRU_64_2": {"class": RNN_GRU, "hidden_size": 64, "num_layers": 2},
        "MyTransformer": {
            "class": MyTransformer,
            "embed_size": 32,
            "output_channel": 4,
            "seq_length": signal_length,
            "num_layers": 3,
            "nhead": 1,
            "forward_expansion": 1,
            "dropout": 0.0,
        },
        "ATT": {
            "class": Encoder,
            "embed_size": 128,
            "output_channel": 4,
            "seq_length": signal_length,
            "num_layers": 7,
            "nhead": 1,
            "forward_expansion": 1,
            "dropout": 0.0,
        },
        "ATT_32_1": {
            "class": Encoder,
            "embed_size": 32,
            "output_channel": 4,
            "seq_length": signal_length,
            "num_layers": 1,
            "nhead": 1,
            "forward_expansion": 2,
            "dropout": 0.0,
        },
        "ATT_64_2": {
            "class": Encoder,
            "embed_size": 64,
            "output_channel": 4,
            "seq_length": signal_length,
            "num_layers": 2,
            "nhead": 4,
            "forward_expansion": 4,
            "dropout": 0.0,
        },
        "FC_196_3": {
            "class": FullyConnectedNet,
            "hidden_dim": 196,
            "num_hidden_layers": 3,
            "output_channel": 4,
            "dim_expand": 0,
        },
        "UNET_24_48_96": {"class": UNET, "features": [24, 48, 96], "output_channel": 4},
        "UNET_32_64_128": {
            "class": UNET,
            "features": [32, 64, 128],
            "output_channel": 4,
        },
        "UNET_4_8_16": {"class": UNET, "features": [4, 8, 16], "output_channel": 4},
    }

    if model_name not in model_configs:
        raise ValueError(f"Model '{model_name}' is not implemented.")

    config = model_configs[model_name]
    if model_name == "MyTransformer":
        model = config["class"](
            feature_channel=feature_channel,
            output_channel=config["output_channel"],
            embed_size=config["embed_size"],
            seq_length=config["seq_length"],
            num_layers=config["num_layers"],
            nhead=config["nhead"],
            forward_expansion=config["forward_expansion"],
            dropout=config["dropout"],
        )
    if model_name.startswith("ATT"):
        model = config["class"](
            feature_channel=feature_channel,
            output_channel=config["output_channel"],
            embed_size=config["embed_size"],
            num_layers=config["num_layers"],
            heads=config["nhead"],
            forward_expansion=config["forward_expansion"],
            seq_length=config["seq_length"],
            dropout=config["dropout"],
        )
    if model_name.startswith("LSTM") or model_name.startswith("GRU"):
        model = config["class"](
            feature_channel=feature_channel,
            output_channel=4,
            hidden_size=config["hidden_size"],
            num_layers=config["num_layers"],
        )
    if model_name.startswith("FC"):
        model = config["class"](
            in_channels=feature_channel,
            out_channels=config["output_channel"],
            num_hidden_layers=config["num_hidden_layers"],
            hidden_dim=config["hidden_dim"],
            signal_length=signal_length,
            dim_expand=config["dim_expand"],
        )
    if model_name.startswith("UNET"):
        model = config["class"](
            feature_channel=feature_channel,
            output_channel=config["output_channel"],
            features=config["features"],
        )
    return model
