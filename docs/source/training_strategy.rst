Training Strategy
=================

Hyperparameters
---------------

Learning Rate
~~~~~~~~~~~~~

.. code-block:: bash

   --learning_rate 0.001

Batch Size
~~~~~~~~~~

.. code-block:: bash

   --batch_size 32
   --tbatch 24  # Temporal batch size

Loss Functions
~~~~~~~~~~~~~~

.. code-block:: bash

   --loss_type huber
   --beta 0.2  # Weight for absorption loss
   --beta_delta 1.0  # For Huber/SmoothL1

Learning Rate Scheduling
------------------------

RTnn uses ReduceLROnPlateau scheduler:
- Factor: 0.5
- Patience: 5 epochs
- Monitors validation loss

Optimizer
---------

Adam optimizer with default parameters:
- β₁ = 0.9
- β₂ = 0.999
- ε = 1e-8
