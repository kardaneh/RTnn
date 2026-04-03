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
   :target: https://kardaneh.github.io/RTnn/
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

Full documentation is available at: https://kardaneh.github.io/RTnn/

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
    from rtnn.evaluate_helper import check_accuracy_evaluate_lsm

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
    rtnn train --data_path ./data --config config.yaml

    # Evaluate a model
    rtnn evaluate --checkpoint model.pth --data ./data/test/

Project Structure
-----------------

::

    rtnn/
    ├── src/
    │   └── rtnn/                 # Main package
    │       ├── __init__.py
    │       ├── version.py
    │       ├── main.py              # Command-line interface
    │       ├── dataset.py           # LSM data preprocessing
    │       ├── evaluater.py         # Evaluation metrics
    │       ├── utils.py             # File utilities
    │       ├── model_utils.py       # Model utilities
    │       ├── model_loader.py      # Model preparation
    │       ├── diagnostics.py       # Visualization
    │       ├── stats.py             # Statistics computation
    │       └── models/              # Model architectures
    │           ├── __init__.py
    │           ├── rnn.py           # LSTM/GRU models
    │           ├── fcn.py           # Fully Connected Network
    │           ├── Transformer.py   # Transformer model
    │           └── DimChangeModule.py
    ├── tests/               # Unit tests (unittest framework)
    ├── docs/                # Documentation
    ├── pyproject.toml
    ├── .pre-commit-config.yaml
    ├── LICENSE
    └── README.rst

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
