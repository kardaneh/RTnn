RTnn
=========================================

|License| |CI Status| |Documentation Status| |Documentation|

RTnn is a PyTorch-based framework for training neural networks to model radiative
transfer processes in climate science, particularly for Land Surface Models (LSM).

.. |License| image:: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-blue.svg
   :target: https://creativecommons.org/licenses/by-nc-sa/4.0/
   :alt: License

.. |CI Status| image:: https://github.com/kardaneh/RTNN/workflows/CI/badge.svg
   :target: https://github.com/kardaneh/RTNN/actions
   :alt: CI Status

.. |Documentation Status| image:: https://github.com/kardaneh/RTNN/workflows/Documentation/badge.svg
   :target: https://github.com/kardaneh/RTNN/actions
   :alt: Documentation Status

.. |Documentation| image:: https://img.shields.io/badge/docs-GitHub%20Pages-blue
   :target: https://rtnn.readthedocs.io/
   :alt: Documentation

Features
--------

* **Multiple Neural Architectures**: LSTM, GRU, Transformer, FCN
* **Climate Data Support**: NetCDF4 format with multi-year and multi-processor handling
* **GPU Acceleration**: CUDA support with multi-GPU training
* **Comprehensive Metrics**: NMAE, NMSE, R², and custom loss functions
* **Visualization Tools**: Built-in plotting for predictions, metrics, and absorption rates
* **CLI Interface**: Easy training and evaluation from command line

Documentation
-------------

Full documentation is available at: https://rtnn.readthedocs.io/

Installation
------------

Using uv (recommended):

.. code-block:: bash

    # Install uv if you haven't already
    pip install uv

    # Clone the repository
    git clone https://github.com/kardaneh/rtnn.git
    cd rtnn

    # Create virtual environment and install
    uv venv
    uv pip install -e .

    # Install with development dependencies
    uv pip install -e ".[dev]"

Quick Start
-----------

.. code-block:: python

    from rtnn import DataPreprocessor, RNN_LSTM
    from rtnn.evaluater import run_validation

    # Check version
    print(f"RTnn version: {rtnn.__version__}")
    print(f"Author: {rtnn.__author__}")

    # Initialize your model
    model = RNN_LSTM(
        feature_channel=6,
        output_channel=4,
        hidden_size=256,
        num_layers=3
    )

    # Prepare your data
    # dataset = DataPreprocessor(...)

    # Train or evaluate...

Command Line Interface
----------------------

.. code-block:: bash

    # Show version
    rtnn --version

    # Show help
    rtnn --help

    # Train a model
    rtnn \\
        --root_dir "./" \\
        --main_folder "Prod__lstm_h256_l3_d0d1_sb_4_ne_100" \\
        --sub_folder "nrm_log1p_standard_lr_0d0001_beta_0d5" \\
        --prefix "nrm_log1p_standard_lr_0d0001_beta_0d5" \\
        --dataset_type "LSM" \\
        --type "lstm" \\
        --hidden_size "256" \\
        --num_layers "3" \\
        --output_channel "120" \\
        --seq_length "10" \\
        --feature_channel "121" \\
        --embed_size "256" \\
        --nhead "4" \\
        --forward_expansion "4" \\
        --dropout "0.1" \\
        --model_name "lstm_h256_l3_d0d1" \\
        --batch_size "4" \\
        --num_epochs "100" \\
        --learning_rate "0.0001" \\
        --loss_type "huber" \\
        --beta "0.5" \\
        --beta_delta "1.0" \\
        --train_data_files "/path/to/training/data" \\
        --test_data_files "/path/to/testing/data" \\
        --train_years "1995-1999" \\
        --test_year "2000" \\
        --norm "log1p_standard" \\
        --num_workers "4" \\
        --save_model "True" \\
        --save_checkpoint_name "model" \\
        --save_per_samples "10000" \\
        --load_checkpoint_name "nrm_log1p_standard_lr_0d0001_beta_0d2_epoch0020_model.pth.tar" \\
        --run_type "train" \\
        --seed "42" \\
        --debug "False"

    # Evaluate a model
    rtnn \\
        --root_dir "./" \\
        --main_folder "Prod__lstm_h256_l3_d0d1_sb_4_ne_100" \\
        --sub_folder "nrm_log1p_standard_lr_0d0001_beta_0d5" \\
        --prefix "nrm_log1p_standard_lr_0d0001_beta_0d5" \\
        --dataset_type "LSM" \\
        --type "lstm" \\
        --hidden_size "256" \\
        --num_layers "3" \\
        --output_channel "120" \\
        --seq_length "10" \\
        --feature_channel "121" \\
        --embed_size "256" \\
        --nhead "4" \\
        --forward_expansion "4" \\
        --dropout "0.1" \\
        --model_name "lstm_h256_l3_d0d1" \\
        --batch_size "4" \\
        --num_epochs "100" \\
        --learning_rate "0.0001" \\
        --loss_type "huber" \\
        --beta "0.5" \\
        --beta_delta "1.0" \\
        --train_data_files "/path/to/training/data" \\
        --test_data_files "/path/to/testing/data" \\
        --train_years "1995-1999" \\
        --test_year "2000" \\
        --norm "log1p_standard" \\
        --num_workers "4" \\
        --save_model "True" \\
        --save_checkpoint_name "model" \\
        --save_per_samples "10000" \\
        --load_checkpoint_name "nrm_log1p_standard_lr_0d0001_beta_0d2_epoch0020_model.pth.tar" \\
        --run_type "inference" \\
        --seed "42" \\
        --debug "False"

Project Structure
-----------------

::

   rtnn/
   ├── src/rtnn/                 # Main package
   │   ├── __init__.py
   │   ├── __main__.py
   │   ├── version.py
   │   ├── dataset.py            # DataPreprocessor class
   │   ├── evaluater.py          # Metrics and loss functions
   │   ├── diagnostics.py        # Visualization tools
   │   ├── logger.py             # Rich logging
   │   ├── main.py               # Main training script
   │   ├── model_loader.py       # Model factory
   │   ├── model_utils.py        # Model utilities
   │   ├── utils.py              # File and general utilities
   │   └── models/               # Model architectures
   │       ├── __init__.py
   │       ├── rnn.py            # LSTM and GRU
   │       ├── fcn.py            # Fully Connected Network
   │       ├── Transformer.py    # Transformer encoder
   │       ├── mlp.py            # Multi-Layer Perceptron
   ├── tests/                    # Unit tests
   │   ├── test_rnn.py
   │   ├── test_fcn.py
   │   ├── test_transformer.py
   │   ├── test_mlp.py
   │   ├── test_dataset.py
   │   ├── test_evaluater.py
   │   ├── test_model_loader.py
   │   ├── test_diagnostics.py
   │   ├── test_utils.py
   │   ├── test_model_utils.py
   │   └── test_runner.py
   ├── docs/                     # Documentation
   │   ├── source/
   │   └── build/
   ├── .github/workflows/        # CI/CD pipelines
   │   ├── ci.yaml
   │   └── docs.yml
   ├── pyproject.toml            # Package configuration
   ├── README.rst                # Project README
   └── LICENSE                   # CC BY-NC-SA 4.0

Dependencies
------------

Core dependencies:
- PyTorch >= 2.0.0
- NumPy
- Xarray
- NetCDF4
- Matplotlib
- scikit-learn
- tqdm
- tensorboard
- pandas
- scipy
- seaborn

For full list, see ``pyproject.toml``.

Testing
-------

RTnn uses Python's built-in `unittest` framework for testing.

Running all tests
~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Run all tests
   python -m unittest discover tests -v

   # Run all tests with the test runner
   python tests/test_runner.py

Running specific model tests
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Run only RNN model tests
   python -m unittest tests.test_rnn -v
   python tests/test_runner.py --pattern test_rnn.py

   # Run only FCN model tests
   python -m unittest tests.test_fcn -v
   python tests/test_runner.py --pattern test_fcn.py

   # Run only Transformer model tests
   python -m unittest tests.test_transformer -v
   python tests/test_runner.py --pattern test_transformer.py

Running specific test classes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Run specific test class
   python -m unittest tests.test_rnn.TestRNN_LSTM
   python -m unittest tests.test_fcn.TestFCN
   python -m unittest tests.test_transformer.TestEncoder

   # Run specific test method
   python -m unittest tests.test_rnn.TestRNN_LSTM.test_forward_shape
   python -m unittest tests.test_fcn.TestFCN.test_forward_shape_default


Development
-----------

This project uses pre-commit hooks for code quality:

.. code-block:: bash

    # Install pre-commit hooks
    pre-commit install

    # Run pre-commit on all files
    pre-commit run --all-files

License
-------

This project is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.

You are free to:
- **Share** — copy and redistribute the material in any medium or format
- **Adapt** — remix, transform, and build upon the material

Under the following terms:
- **Attribution** — You must give appropriate credit, provide a link to the license, and indicate if changes were made.
- **NonCommercial** — You may not use the material for commercial purposes.
- **ShareAlike** — If you remix, transform, or build upon the material, you must distribute your contributions under the same license.

See the `LICENSE` file for full details.

Author
------

**Kazem Ardaneh**
CNRS / IPSL / Sorbonne University
Email: kardaneh@ipsl.fr

Citation
--------

If you use RTnn in your research, please cite:

.. code-block:: bibtex

    @software{ardaneh2026rtnn,
        author = {Kazem Ardaneh},
        title = {RTnn: Radiative Transfer Neural Networks for Climate Science},
        year = {2026},
        publisher = {GitHub},
        url = {https://github.com/kardaneh/rtnn}
    }
