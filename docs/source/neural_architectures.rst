Neural Architectures
====================

RTnn supports multiple neural network architectures for radiative transfer modeling.

Recurrent Neural Networks (RNN)
-------------------------------

LSTM (Long Short-Term Memory)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Bidirectional LSTM with final Conv1d projection.

.. code-block:: python

   from rtnn import RNN_LSTM

   model = RNN_LSTM(
       feature_channel=6,    # Input features
       output_channel=4,     # Output channels
       hidden_size=128,      # Hidden state size
       num_layers=3          # Number of LSTM layers
   )

**Architecture:**
- Bidirectional LSTM: captures forward and backward dependencies
- Conv1d output: projects hidden states to output channels

GRU (Gated Recurrent Unit)
~~~~~~~~~~~~~~~~~~~~~~~~~~

Similar to LSTM but with fewer parameters.

.. code-block:: python

   from rtnn import RNN_GRU

   model = RNN_GRU(
       feature_channel=6,
       output_channel=4,
       hidden_size=128,
       num_layers=3
   )

Transformer
-----------

Self-attention based encoder for sequence processing.

.. code-block:: python

   from rtnn import TransformerEncoder

   model = TransformerEncoder(
       feature_channel=6,
       output_channel=4,
       embed_size=64,        # Embedding dimension
       num_layers=2,         # Number of transformer blocks
       heads=4,              # Attention heads
       forward_expansion=4,  # Feed-forward expansion factor
       seq_length=10,        # Input sequence length
       dropout=0.1           # Dropout rate
   )

**Features:**
- Positional embeddings
- Multi-head self-attention
- Residual connections
- Layer normalization

FCN (Fully Connected Network)
-----------------------------

Deep fully connected network with batch normalization.

.. code-block:: python

   from rtnn import FCN

   model = FCN(
       feature_channel=6,
       output_channel=4,
       num_layers=3,         # Number of hidden layers
       hidden_size=196,      # Hidden layer size
       seq_length=10,        # Input sequence length
       dim_expand=0          # Optional sequence expansion
   )

**Architecture:**
- Flattens input: (batch, channels, seq) → (batch, channels * seq)
- FCBlock: Linear → BatchNorm → ReLU
- Optional sequence length expansion

UNet1D
------

1D U-Net architecture for sequence-to-sequence tasks.

.. code-block:: python

   from rtnn import UNET

   model = UNET(
       feature_channel=6,
       output_channel=4,
       features=[32, 64, 128]  # Channel progression
   )

**Features:**
- Encoder-decoder with skip connections
- Max pooling for downsampling
- Transposed convolutions for upsampling
- Preserves sequence length

Model Comparison
----------------

.. list-table::
   :widths: 20 25 25 30
   :header-rows: 1

   * - Architecture
     - Parameters
     - Best For
     - Pros/Cons
   * - LSTM/GRU
     - Moderate
     - Temporal dependencies
     - Good for sequences, can be slow
   * - Transformer
     - Large
     - Long-range dependencies
     - Parallel processing, memory intensive
   * - FCN
     - Moderate
     - Simple relationships
     - Fast, no temporal modeling
   * - UNet
     - Moderate
     - Local patterns
     - Good for spatial features

Choosing an Architecture
------------------------

1. **LSTM/GRU**: Default choice for temporal sequences
2. **Transformer**: For long sequences or when attention is important
3. **FCN**: For simple regression tasks without temporal structure
4. **UNet**: For local pattern recognition

See :doc:`training_strategy` for hyperparameter recommendations.
