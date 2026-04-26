# Copyright 2026 IPSL / CNRS / Sorbonne University
# Authors: Kazem Ardaneh
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

from rtnn.models.rnn import RNN_LSTM, RNN_GRU
from rtnn.models.Transformer import Encoder
from rtnn.models.Transformer import EncoderTorch
from rtnn.models.fcn import FCN
from rtnn.models.fcn import VerticalRTColumnNet


def load_model(args):
    """
    Load and initialize a model based on the provided configuration.

    This function acts as a factory that instantiates the appropriate model
    architecture based on the `type` argument. Supported models include:
    - LSTM: Bidirectional LSTM with Conv1d output projection
    - GRU: Bidirectional GRU with Conv1d output projection
    - Transformer: Transformer encoder with positional embeddings
    - FCN/fullyconnected: Fully connected network with configurable depth

    Parameters
    ----------
    args : argparse.Namespace
        Namespace containing model configuration parameters. Required attributes
        depend on the model type:

        For LSTM/GRU:
            - type : str ('lstm' or 'gru')
            - feature_channel : int
            - output_channel : int
            - hidden_size : int
            - num_layers : int

        For Transformer:
            - type : str ('transformer')
            - feature_channel : int
            - output_channel : int
            - embed_size : int
            - num_layers : int
            - nhead : int
            - forward_expansion : int
            - seq_length : int
            - dropout : float

        For FCN/fullyconnected:
            - type : str ('fcn' or 'fullyconnected')
            - feature_channel : int
            - output_channel : int
            - num_layers : int
            - hidden_size : int
            - seq_length : int
            - dim_expand : int (optional, default 0)

    Returns
    -------
    torch.nn.Module
        Initialized PyTorch model of the specified architecture.

    Raises
    ------
    ValueError
        If the specified model type is not implemented.

    Examples
    --------
    >>> args = argparse.Namespace(
    ...     type='lstm',
    ...     feature_channel=6,
    ...     output_channel=4,
    ...     hidden_size=128,
    ...     num_layers=3
    ... )
    >>> model = load_model(args)
    >>> print(type(model))
    <class 'rtnn.models.rnn.RNN_LSTM'>

    >>> args = argparse.Namespace(
    ...     type='transformer',
    ...     feature_channel=6,
    ...     output_channel=4,
    ...     embed_size=64,
    ...     num_layers=2,
    ...     nhead=4,
    ...     forward_expansion=4,
    ...     seq_length=10,
    ...     dropout=0.1
    ... )
    >>> model = load_model(args)
    >>> print(type(model))
    <class 'rtnn.models.Transformer.Encoder'>

    >>> args = argparse.Namespace(
    ...     type='fcn',
    ...     feature_channel=6,
    ...     output_channel=4,
    ...     num_layers=3,
    ...     hidden_size=196,
    ...     seq_length=10
    ... )
    >>> model = load_model(args)
    >>> print(type(model))
    <class 'rtnn.models.fcn.FCN'>
    """
    model_type = args.type.lower()

    if model_type in ["lstm", "gru"]:
        model_class = RNN_LSTM if model_type == "lstm" else RNN_GRU
        model = model_class(
            feature_channel=args.feature_channel,
            output_channel=args.output_channel,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
        )

    elif model_type == "transformer":
        model = Encoder(
            feature_channel=args.feature_channel,
            output_channel=args.output_channel,
            embed_size=args.embed_size,
            num_layers=args.num_layers,
            heads=args.nhead,
            forward_expansion=args.forward_expansion
            if args.forward_expansion is not None
            else 1,
            seq_length=args.seq_length,
            dropout=args.dropout,
        )

    elif model_type == "encodertorch":
        # New EncoderTorch implementation (PyTorch native transformer)
        model = EncoderTorch(
            feature_channel=args.feature_channel,
            output_channel=args.output_channel,
            embed_size=args.embed_size,
            num_layers=args.num_layers,
            heads=args.nhead,
            forward_expansion=args.forward_expansion
            if args.forward_expansion is not None
            else 4,  # Default expansion factor
            seq_length=args.seq_length,
            dropout=args.dropout,
        )

    elif model_type in ["vrtn", "verticalrt", "vertical"]:
        model = VerticalRTColumnNet(
            feature_channel=args.feature_channel,
            hidden=args.hidden_size,
            out_channel=args.output_channel,
            n_layers=args.seq_length,
        )

    elif model_type in ["fcn", "fullyconnected"]:
        model = FCN(
            feature_channel=args.feature_channel,
            output_channel=args.output_channel,
            num_layers=args.num_layers,
            hidden_size=args.hidden_size,
            seq_length=args.seq_length,
            dim_expand=0,
        )

    else:
        raise ValueError(f"Model type '{args.type}' is not implemented.")

    return model
