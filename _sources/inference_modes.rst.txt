Inference Modes
===============

Running inference with trained models.

Command Line
------------

Set `--run_type inference` to run inference without training:

.. code-block:: bash

   rtnn \\
       --run_type inference \\
       --load_checkpoint_name model.pth.tar \\
       --type verticalrt \\
       --feature_channel 121 \\
       --output_channel 120 \\
       --hidden_size 256 \\
       --num_layers 3 \\
       --seq_length 10 \\
       --test_year 2000 \\
       --test_data_files /path/to/data

**Key inference arguments:**

- `--run_type inference`: Enables inference mode (no training loop)
- `--load_checkpoint_name`: Path to the trained model checkpoint
- `--save_model False`: Disables checkpoint saving (default for inference)
- `--num_workers 0`: Single worker for deterministic inference

SBATCH Script for Inference
---------------------------

.. code-block:: bash

   #SBATCH --partition=batch
   #SBATCH --gpus=1
   #SBATCH --ntasks-per-node=1
   #SBATCH --cpus-per-task=4
   #SBATCH --mem=48G
   #SBATCH --time=1:00:00

   source .venv/bin/activate

   rtnn \\
       --run_type inference \\
       --load_checkpoint_name model.pth.tar \\
       --type verticalrt \\
       --feature_channel 121 \\
       --output_channel 120 \\
       --test_year 2000 \\
       --test_data_files /data/path \\
       --num_workers 0

Loading Checkpoints
-------------------

RTnn saves two types of files:

- `.pth.tar`: Checkpoint with model, optimizer, and training history
- `.pth`: Full model only

To load a checkpoint for inference:

.. code-block:: bash

   # Use --load_checkpoint_name with .pth.tar file
   rtnn --run_type inference --load_checkpoint_name model.pth.tar ...

Resume Training
---------------

To resume training from a checkpoint:

.. code-block:: bash

   rtnn \\
       --run_type resume_train \\
       --load_checkpoint_name model.pth.tar \\
       --num_epochs 100

Inference Performance Tips
--------------------------

1. Use `--num_workers 0` for deterministic inference
2. Reduce batch size if memory constrained (4-8 for 120 output channels)
3. Ensure `--save_model False` to avoid writing checkpoints during inference

See :doc:`training_strategy` for training configuration.
