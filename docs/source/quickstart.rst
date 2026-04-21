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

Performance Optimization
------------------------------------

For optimal performance, different configurations of SLURM parameters
and PyTorch DataLoader workers were tested. The following table summarizes the results.
Note that `ntasks-per-node × cpus-per-task` is set to a maximum of 4.

.. list-table:: Performance Comparison on HAL Machine
   :header-rows: 1
   :widths: 5 15 15 15 25 15 15 15
   :align: center

   * - Test
     - ntasks-per-node
     - cpus-per-task
     - num_workers
     - SBATCH lines
     - Epoch 0 Time (s)
     - Epoch 1 Time (s)
     - Total Time (s)
   * - 1
     - default (1)
     - default
     - 0
     - (omit both)
     - 1179.8
     - 742.2
     - 1922.0
   * - 2
     - default (1)
     - default
     - 4
     - (omit both)
     - 333.5
     - 280.7
     - 614.2
   * - 3
     - default (1)
     - 4
     - 0
     - #SBATCH --cpus-per-task=4
     - 1150.6
     - 997.1
     - 2147.7
   * - 4
     - default (1)
     - 4
     - 4
     - #SBATCH --cpus-per-task=4
     - 375.5
     - 286.8
     - 662.3
   * - 5
     - 4
     - default
     - 0
     - #SBATCH --ntasks-per-node=4
     - 1898.0
     - (Missing)
     - ~1898.0+
   * - 6
     - 4
     - default
     - 4
     - #SBATCH --ntasks-per-node=4
     - 389.5
     - 283.6
     - 673.1
   * - 7
     - 2
     - 2
     - 0
     - Both lines
     - 1114.6
     - 753.2
     - 1867.8
   * - 8
     - 2
     - 2
     - 4
     - Both lines
     - 338.6
     - 275.9
     - 614.5

**Configuration Notes:**

- ``default`` for ntasks-per-node means 1 (system default)
- ``default`` for cpus-per-task means the parameter is omitted (system default)
- Epoch 0 includes initial data loading, cache warm-up, and JIT compilation
- Epoch 1 shows steady-state performance after optimization
- Maximum total cores constraint: `ntasks-per-node × cpus-per-task ≤ 4`
- Test 5 missing epoch 1 due to job timeout (I/O bottleneck)

**Performance Analysis and Key Findings:**

1. **Dramatic Impact of DataLoader Workers**:
   - Adding `num_workers=4` reduces total time by **68%** (Test 1→2: 1922s → 614s)
   - Without workers, all configurations perform poorly (1867-2148s total time)
   - Workers effectively overlap I/O with GPU computation

2. **Optimal Configuration: Test 2 and Test 8**:
   - **Test 2** (default/4 workers): 614.2s total, 280.7s epoch 1
   - **Test 8** (ntasks=2, cpus=2, workers=4): 614.5s total, 275.9s epoch 1
   - Both achieve **3.1x speedup** over baseline (Test 1)
   - Simple configuration (Test 2) is easiest to implement

3. **Poor Configurations to Avoid**:
   - **Test 5** (ntasks=4, no workers): 1898s epoch 0, incomplete - I/O saturation causes timeout
   - **Test 3 & 7** (workers=0): 1867-2148s total time - CPU cores wasted without workers
   - Adding CPU cores without workers provides **no benefit** (Test 1 vs 3 vs 7)

4. **The Workers Effect**:
   - With workers (Tests 2,4,6,8): Epoch 0: 333-390s, Epoch 1: 276-287s
   - Without workers (Tests 1,3,5,7): Epoch 0: 1115-1898s, Epoch 1: 742-997s
   - Workers reduce epoch 0 time by **70-80%** and epoch 1 time by **62-71%**

5. **CPU Core Allocation Impact**:
   - With workers, CPU core allocation has minimal effect (614-673s total)
   - Without workers, more cores actually hurt performance (Test 5: 1898s vs Test 1: 1922s)
   - Suggests I/O is the primary bottleneck, not CPU compute


**Best Configuration (Simple and Effective):**

.. code-block:: bash

   #SBATCH --ntasks-per-node=1
   #SBATCH --cpus-per-task=1

   # In your training script:
   num_workers=4

Next Steps
----------

- Explore :doc:`neural_architectures` for different model types
- Learn about :doc:`training_strategy` for optimal training
- Check :doc:`inference_modes` for running predictions
- See :doc:`api/modules` for detailed API reference
