Quick Start
===========

This guide will help you get started with RTnn quickly.

Basic Usage
-----------

.. code-block:: python

   from rtnn import DataPreprocessor, RNN_LSTM
   from rtnn.logger import Logger

   # Initialize logger
   logger = Logger(console_output=True)

   # Load your data
   dataset = DataPreprocessor(
       logger=logger,
       dfs=["data_1995.nc", "data_1996.nc"],
       stime=0,
       tstep=100,
       tbatch=24,
       norm_mapping=norm_mapping,
       normalization_type=normalization_type
   )

   # Create model
   model = RNN_LSTM(
       feature_channel=6,
       output_channel=4,
       hidden_size=128,
       num_layers=3
   )

   # Train (simplified)
   for epoch in range(num_epochs):
       for features, targets in dataloader:
           outputs = model(features)
           loss = criterion(outputs, targets)
           loss.backward()
           optimizer.step()

Command Line Interface
----------------------

Train a model:

.. code-block:: bash

   rtnn \\
       --type lstm \\
       --hidden_size 128 \\
       --num_layers 3 \\
       --batch_size 32 \\
       --num_epochs 100 \\
       --learning_rate 0.001 \\
       --train_years "1995-1999" \\
       --test_year 2000 \\
       --main_folder results \\
       --sub_folder experiment_1

Show version:

.. code-block:: bash

   rtnn --version

Show help:

.. code-block:: bash

   rtnn --help

Example: Training an LSTM Model
-------------------------------

Here's a complete example using the SBATCH script:

.. code-block:: bash

   #!/usr/bin/env bash
   #SBATCH -A your_account
   #SBATCH -p boost_usr_prod
   #SBATCH --qos=boost_qos_dbg
   #SBATCH --time=00:30:00
   #SBATCH -N 1
   #SBATCH --gpus-per-node=4

   module load profile/deeplrn
   module load cineca-ai
   source .venv/bin/activate

   rtnn \\
       --type lstm \\
       --hidden_size 64 \\
       --num_layers 2 \\
       --batch_size 16 \\
       --num_epochs 2 \\
       --learning_rate 0.0001 \\
       --loss_type huber \\
       --train_years 1998 \\
       --test_year 1999 \\
       --main_folder Debug__lstm_h64_l2_sb_16_ne_2 \\
       --sub_folder run_$(date +"%Y%m%d_%H%M%S")

Next Steps
----------

- Explore :doc:`neural_architectures` for different model types
- Learn about :doc:`training_strategy` for optimal training
- Check :doc:`inference_modes` for running predictions
- See :doc:`api/modules` for detailed API reference
