Project Structure
=================

The RTnn package is organized as follows:

.. code-block:: text

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
   │   ├── stats.py              # Statistical computations
   │   └── models/               # Model architectures
   │       ├── __init__.py
   │       ├── rnn.py            # LSTM and GRU
   │       ├── fcn.py            # Fully Connected Network
   │       ├── Transformer.py    # Transformer encoder
   │       ├── UNet1D.py         # 1D U-Net
   │       └── DimChangeModule.py
   ├── tests/                    # Unit tests
   │   ├── test_rnn.py
   │   ├── test_fcn.py
   │   ├── test_transformer.py
   │   ├── test_dataset.py
   │   ├── test_evaluater.py
   │   ├── test_model_loader.py
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

Module Descriptions
-------------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Module
     - Description
   * - ``dataset.py``
     - DataPreprocessor for loading and preprocessing NetCDF files
   * - ``evaluater.py``
     - Loss functions and evaluation metrics
   * - ``diagnostics.py``
     - Plotting and visualization utilities
   * - ``logger.py``
     - Rich console and file logging
   * - ``main.py``
     - Main training pipeline entry point
   * - ``model_loader.py``
     - Factory function for model instantiation
   * - ``model_utils.py``
     - Model inspection and checkpoint management
   * - ``utils.py``
     - File utilities and EasyDict class
   * - ``stats.py``
     - Statistical computation for normalization
   * - ``models/``
     - Neural network architectures
