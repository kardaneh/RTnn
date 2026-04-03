RTnn: Radiative Transfer Neural Networks
=========================================

CI, and Documentation Status, License, Python and PyTorch versions
------------------------------------------------------------------

.. image:: https://github.com/kardaneh/RTNN/workflows/CI/badge.svg
   :target: https://github.com/kardaneh/RTNN/actions
   :alt: CI Status

.. image:: https://github.com/kardaneh/RTNN/workflows/Documentation/badge.svg
   :target: https://github.com/kardaneh/RTNN/actions
   :alt: Documentation Status

.. image:: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-blue.svg
   :target: https://creativecommons.org/licenses/by-nc-sa/4.0/
   :alt: License

.. image:: https://img.shields.io/badge/python-3.9+-blue.svg
   :alt: Python Version

.. image:: https://img.shields.io/badge/PyTorch-2.0+-red.svg
   :alt: PyTorch

RTnn is a PyTorch-based framework for training neural networks to model radiative
transfer processes in climate science, particularly for Land Surface Models (LSM).


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
    │       ├── cli.py           # Command-line interface
    │       ├── data_helper_lsm.py   # LSM data preprocessing
    │       ├── evaluate_helper.py   # Evaluation metrics
    │       ├── file_helper.py       # File utilities
    │       ├── model_helper.py      # Model utilities
    │       ├── model_prepare.py     # Model preparation
    │       ├── plot_helper.py       # Visualization
    │       ├── stats.py             # Statistics computation
    │       └── models/              # Model architectures
    │           ├── __init__.py
    │           ├── rnn.py           # LSTM/GRU models
    │           ├── fcn.py           # Fully Connected Network
    │           ├── Transformer.py   # Transformer model
    │           ├── UNet1D.py        # 1D UNet model
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

Run the test suite using unittest:

.. code-block:: bash

    # Run all tests
    python -m unittest discover tests -v

    # Run specific test file
    python -m unittest tests.test_models

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

Project Links
-------------

* Source Code: https://github.com/kardaneh/rtnn
* Issue Tracker: https://github.com/kardaneh/rtnn/issues
