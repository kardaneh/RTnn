Inference Modes
===============

Running inference with trained models.

Command Line
------------

Set `--run_type inference` to run inference without training:

.. code-block:: bash

      rtnn \
        --root_dir "./" \  # Project root directory
        --main_folder "Prod__lstm_h256_l3_d0d1_sb_4_ne_100" \  # Main experiment folder
        --sub_folder "nrm_log1p_standard_lr_0d0001_beta_0d5" \  # Run-specific subfolder
        --prefix "nrm_log1p_standard_lr_0d0001_beta_0d5" \  # Output/checkpoint prefix
        --dataset_type "LSM" \  # Dataset type
        --type "lstm" \  # Model type
        --hidden_size "256" \  # Hidden layer size
        --num_layers "3" \  # Number of layers
        --output_channel "120" \  # Output feature dimension
        --seq_length "10" \  # Input sequence length
        --feature_channel "121" \  # Input feature dimension
        --embed_size "256" \  # Embedding size
        --nhead "4" \  # Number of attention heads (if applicable)
        --forward_expansion "4" \  # Feed-forward expansion factor
        --dropout "0.1" \  # Dropout rate
        --model_name "lstm_h256_l3_d0d1" \  # Model identifier
        --batch_size "4" \  # Batch size
        --num_epochs "100" \  # Number of epochs (used for config consistency)
        --learning_rate "0.0001" \  # Learning rate
        --loss_type "huber" \  # Loss function
        --beta "0.5" \  # Loss weighting parameter
        --beta_delta "1.0" \  # Secondary loss scaling factor
        --train_data_files "/path/to/training/data" \  # Training data path (optional for inference context)
        --test_data_files "/path/to/testing/data" \  # Test data path
        --train_years "1995-1999" \  # Training time range
        --test_year "2000" \  # Test year
        --norm "log1p_standard" \  # Normalization method
        --num_workers "4" \  # DataLoader workers
        --save_model "True" \  # Save outputs
        --save_checkpoint_name "model" \  # Output checkpoint name
        --save_per_samples "10000" \  # Save interval
        --load_checkpoint_name "nrm_log1p_standard_lr_0d0001_beta_0d2_epoch0020_model.pth.tar" \  # Model checkpoint to load
        --run_type "inference" \  # Run mode: inference
        --seed "42" \  # Random seed
        --debug "False"  # Debug mode

**Key inference arguments:**

- `--run_type inference`: Enables inference mode (no training loop)
- `--load_checkpoint_name`: Path to the trained model checkpoint
- `--save_model False`: Disables checkpoint saving (default for inference)
- `--num_workers 0`: Single worker for deterministic inference


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
