# Copyright 2026 IPSL / CNRS / Sorbonne University
# Authors: Kazem Ardaneh
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

"""
rtnn.model_loader - Model Factory for Radiative Transfer Neural Networks

This module provides a factory function `load_model()` that instantiates
various neural network architectures for radiative transfer calculations
in climate modeling. It serves as the central point for model creation
across the RTnn framework.

Module Overview
---------------
The model loader module implements a factory pattern to abstract away the
details of model instantiation. Based on a configuration object (typically
from command-line arguments), it returns an initialized PyTorch model ready
for training or inference.

Supported Model Architectures
-----------------------------
1. **Recurrent Neural Networks (RNN)**

   - LSTM (Long Short-Term Memory): Bidirectional LSTM with Conv1d projection
   - GRU (Gated Recurrent Unit): Bidirectional GRU with Conv1d projection
   - Best for: Sequential data with temporal dependencies

2. **Transformer Models**

   - EncoderTorch: PyTorch-native transformer encoder
   - Features: Multi-head self-attention, positional embeddings
   - Best for: Long-range dependencies and parallel processing

3. **Fully Connected Networks (FCN)**

   - Standard FCN: Multi-layer perceptron with configurable depth
   - PINN: Physics-Informed Neural Network with two-stream architecture for vertical profiles
   - Best for: Non-sequential, feature-based inputs

4. **Multi-Layer Perceptrons (MLP)**

   - MLP: Standard MLP with batch/layer normalization options
   - MLPResidual: MLP with residual connections between layers
   - Features: Positional embeddings, dropout, multiple activation functions
   - Best for: Flexible architecture with skip connections

Architecture Details
--------------------
All models are designed to handle the specific requirements of radiative
transfer problems:
- Input shape: (batch, seq_len, feature_channel)
- Output shape: (batch, seq_len, output_channel)
- Physical constraints: Conservation of energy, positive outputs

The models output four channels corresponding to:
- collim_alb (collimated albedo)
- collim_tran (collimated transmittance)
- isotrop_alb (isotropic albedo)
- isotrop_tran (isotropic transmittance)

Data Flow
---------
1. Parse configuration (args)
2. Determine model type from args.type
3. Extract model-specific parameters
4. Instantiate appropriate model class
5. Return initialized model
"""

from rtnn.models.rnn import RNN_LSTM, RNN_GRU
from rtnn.models.transformer import EncoderTorch
from rtnn.models.fcn import FCN
from rtnn.models.pinn import PINN
from rtnn.models.mlp import MLP, MLPResidual


def load_model(args):
    """
    Factory function to instantiate a neural network model from configuration.

    This function builds and returns a PyTorch model based on the value of
    ``args.type``. It supports multiple architectures including recurrent,
    convolutional, transformer-based, and fully connected models.

    Supported model families
    ------------------------
    - LSTM / GRU: Bidirectional recurrent models with Conv1d projection head
    - Transformer: PyTorch Transformer encoder with positional embeddings
    - FCN / FullyConnected: Fully connected feedforward network for sequences
    - VRT / VerticalRT: Physics-inspired vertical column model
    - MLP: Flexible multilayer perceptron with optional embeddings and residuals
    - MLPResidual: Deep residual MLP with layer-wise skip connections

    Returns
    -------
    torch.nn.Module
        Instantiated PyTorch model corresponding to the requested architecture.

    Raises
    ------
    ValueError
        If ``args.type`` does not match any supported model.

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

    elif model_type in ["encodertorch", "transformer"]:
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

    elif model_type in ["pinn"]:
        model = PINN(
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

    elif model_type in ["mlp"]:
        # Standard MLP with configurable hidden layers
        hidden_sizes = getattr(args, "hidden_sizes", [512, 256, 128])
        if isinstance(hidden_sizes, str):
            hidden_sizes = [int(x) for x in hidden_sizes.split(",")]

        model = MLP(
            feature_channel=args.feature_channel,
            output_channel=args.output_channel,
            seq_length=args.seq_length,
            hidden_sizes=hidden_sizes,
            dropout=getattr(args, "dropout", 0.1),
            use_batch_norm=getattr(args, "use_batch_norm", True),
            use_layer_norm=getattr(args, "use_layer_norm", False),
            use_residual=getattr(args, "use_residual", False),
            activation=getattr(args, "activation", "relu"),
            use_positional_embedding=getattr(args, "use_positional_embedding", True),
            positional_embed_dim=getattr(args, "positional_embed_dim", 16),
        )

    elif model_type in ["mlp_residual"]:
        # MLP with residual connections between layers
        model = MLPResidual(
            feature_channel=args.feature_channel,
            output_channel=args.output_channel,
            seq_length=args.seq_length,
            hidden_size=getattr(args, "hidden_size", 256),
            num_layers=getattr(args, "num_layers", 4),
            dropout=getattr(args, "dropout", 0.1),
        )

    else:
        raise ValueError(f"Model type '{args.type}' is not implemented.")

    return model
