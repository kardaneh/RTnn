Neural Architectures
====================

RTnn supports multiple neural network architectures for radiative transfer emulation.

Recurrent Neural Networks (RNN)
-------------------------------

LSTM (Long Short-Term Memory)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Bidirectional LSTM with final Conv1d projection.

.. code-block:: python

   from rtnn import RNN_LSTM

   model = RNN_LSTM(
       feature_channel=121,    # Input features
       output_channel=120,     # Output channels
       hidden_size=256,      # Hidden state size
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
       feature_channel=121,
       output_channel=120,
       hidden_size=256,
       num_layers=3
   )

Transformer
-----------

Self-attention based encoder for sequence processing.

.. code-block:: python

   from rtnn import TransformerEncoder

   model = TransformerEncoder(
       feature_channel=121,
       output_channel=120,
       embed_size=256,        # Embedding dimension
       num_layers=3,         # Number of transformer blocks
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
       feature_channel=121,
       output_channel=120,
       num_layers=3,         # Number of hidden layers
       hidden_size=256,      # Hidden layer size
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
       dropout=0.1            # Dropout rate
   )

Physical Interpretation
-----------------------

This model is a discrete, layer-wise approximation of the two-stream radiative
transfer system, where upward and downward fluxes are coupled at each canopy layer.

Instead of solving a closed-form continuous equation, the network learns the
iterative propagation:

.. math::

   d_l = T_l^{\downarrow} \, d_{l-1} + C_l^{\downarrow} \, u_{l-1} + S_l^{\downarrow}

   u_l = T_l^{\uparrow} \, u_{l-1} + C_l^{\uparrow} \, d_{l-1} + S_l^{\uparrow}

where:

- :math:`d_l` is the downward flux at layer :math:`l`
- :math:`u_l` is the upward flux at layer :math:`l`
- :math:`T^{\downarrow}, T^{\uparrow} \in (0,1)` are learned transmittance terms
- :math:`C^{\downarrow}, C^{\uparrow} \in (0,1)` are coupling (scattering) terms
- :math:`S^{\downarrow}, S^{\uparrow}` are source terms (direct + diffuse forcing)

Surface Boundary Condition
--------------------------

At the bottom of the canopy, the upward flux is initialized using a learned
surface reflection operator:

.. math::

   u_{L-1} = f_{\text{surface}}(d_{L-1})

This corresponds to:

- reflection of downward flux at the surface
- implicit dependence on surface albedo encoded in features

Upward Sweep Refinement
-----------------------

After computing downward fluxes, the upward pass refines radiation propagation:

.. math::

   u_l = T_l^{\uparrow} \, u_{l+1} + C_l^{\uparrow} \, d_l + S_l^{\uparrow}

Flux Reconstruction
-------------------

The final radiative output at each layer is computed as:

.. math::

   y_l = f_D(d_l) + f_U(u_l) + f_{\text{skip}}(h_l)

where:

- :math:`f_D` = downward projection head
- :math:`f_U` = upward projection head
- :math:`f_{\text{skip}}` = residual connection from encoder state :math:`h_l`

Interpretation
--------------

This formulation explicitly mirrors the Python implementation:

- Sequential **downward recurrence** (top → bottom)
- Surface-driven initialization of upward flux
- Sequential **upward recurrence** (bottom → top)
- Final per-layer mixing of both flux streams

This structure preserves:

- Energy exchange between streams (via coupling terms)
- Vertical dependency of canopy radiative transfer
- Physical consistency with two-stream discrete RT solvers

-  See :doc:`training_strategy` for hyperparameter recommendations.
-  See :doc:`experiments` for performance comparisons between architectures.
