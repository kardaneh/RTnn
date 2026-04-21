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

Vertical RT Column Network (Physics-Inspired)
---------------------------------------------

A physics-inspired neural network that emulates the two-stream matrix-based
radiative transfer solver. This architecture preserves the physical structure
of the RT equations, making it particularly well-suited for vertical canopy
radiative transfer modeling.

.. code-block:: python

   from rtnn import VerticalRTColumnNet

   model = VerticalRTColumnNet(
       feature_channel=121,   # Input features (cosz + LAI + SSA + RS)
       hidden=256,            # Hidden dimension size
       out_channel=120,       # Output channels (4 vars × 15 PFTs × 2 bands)
       n_layers=10,           # Number of vertical canopy layers
       layer_embed_dim=16,    # Dimension of layer positional embedding
       mu_bar=0.5,            # Average inverse diffuse optical depth
       dropout=0.1            # Dropout rate
   )

**Physical Interpretation:**

The architecture follows the analytical solution of the two-stream equations:

.. math::

   F(\tau) = A \cdot e^{\lambda \tau} + B \cdot e^{-\lambda \tau} + C \cdot e^{-\tau / \mu_0}

Where:

- :math:`\lambda = \sqrt{\gamma_1^2 - \gamma_2^2}` (eigenvalue)
- :math:`\Gamma = \gamma_2 / (\gamma_1 + \lambda)` (reflection/transmission ratio)
- :math:`C^+` and :math:`C^-` are particular solutions for direct beam

**Key Components:**

1. **Layer Positional Embedding**: Adds vertical position information (depth awareness)

2. **Optical Properties Predictor**: Computes per-layer:

   - :math:`\omega` (single scattering albedo)
   - :math:`\gamma_1, \gamma_2, \gamma_3, \gamma_4` (scattering coefficients)
   - :math:`K = G(\mu)/\mu` (extinction coefficient for direct beam)

3. **Downward Sweep** (Top → Bottom):

   - Models downward diffuse flux propagation through the canopy

4. **Surface Boundary Condition**:

   - Reflects bottom flux based on surface albedo :math:`R_s`

5. **Upward Sweep** (Bottom → Top):

   - Models upward diffuse flux propagation

6. **Flux Reconstruction**:

   - Combines downward and upward fluxes with residual connections

**Advantages:**

- Preserves physical structure of RT equations
- Handles vertical heterogeneity naturally
- Learns layer-specific optical properties
- Physically interpretable parameters
- Efficient for multi-layer canopy problems

**When to Use:**

- Modeling radiative transfer through vegetation canopies
- Problems with vertical structure (multiple layers)
- When physical interpretability is important
- Emulating two-stream RT solvers in Land Surface Models

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
   * - VerticalRT
     - Moderate-Large
     - Vertical RT problems
     - Physics-inspired, interpretable, specialized

Choosing an Architecture
------------------------

1. **LSTM/GRU**: Default choice for temporal sequences
2. **Transformer**: For long sequences or when attention is important
3. **FCN**: For simple regression tasks without temporal structure
4. **VerticalRT**: For vertical canopy radiative transfer problems (physics-inspired)

See :doc:`training_strategy` for hyperparameter recommendations.
