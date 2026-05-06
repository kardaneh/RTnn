RTnn
=========================================

|License| |CI Status| |Documentation Status| |Documentation|

RTnn is a PyTorch library for training neural networks to model radiative transfer in climate science.
RTnn has been developed in the context of the **AI4PEX project** (Research Focus LAND). AI4PEX is focused on enhancing our understanding of how terrestrial ecosystems respond to climate change and the feedback of increased atmospheric CO2 levels to the climate system. The project aims to reduce uncertainties and enhance process representation, namely:

- **Hybrid Modelling and History Matching**: to better predict the instantaneous vegetation responses to water and heat stress.
- **Leverage Deep Learning**: approaches, such as Long-Short Term Memory networks, to simulate phenology and enhance online deep learning frameworks to represent plant carbon dynamics and explore tree mortality drivers.
- **Temperature Sensitivity of Decomposition**: Address the challenge of understanding how temperature affects soil decomposition, which is crucial for ecosystem carbon turnover and land-atmosphere carbon responses to warming.
- **Land-Atmosphere Feedbacks**: Improve the representation of processes that control energy feedbacks to the atmosphere, including regional climate extremes and land carbon uptake, to reduce uncertainties in projected warming trends.

By focusing on these areas, AI4PEX aims to provide a more accurate representation of ecosystem dynamics and feedbacks in climate models.

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

* **Multiple Neural Architectures**: LSTM, GRU, Transformer, FCN, MLP
* **Climate Data Support**: NetCDF4 format with multi-year and multi-processor handling
* **GPU Acceleration**: CUDA support with multi-GPU training
* **Comprehensive Metrics**: NMAE, NMSE, R², and custom loss functions
* **Visualization Tools**: Built-in plotting for predictions, metrics, absorption rates, etc.
* **CLI Interface**: Easy command-line training and evaluation

Documentation
-------------

Full documentation is available at: https://rtnn.readthedocs.io/

Installation
------------

Using uv (recommended):

.. code-block:: bash

    # Install uv (via pip or curl)
    pip install uv
    # or (Linux/macOS)
    curl -LsSf https://astral.sh/uv/install.sh | sh
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc

    # Clone the repository
    git clone https://github.com/kardaneh/rtnn.git
    cd rtnn

    # Create virtual environment and activate
    uv venv --python 3.8
    source .venv/bin/activate

    # Install RTnn in editable mode
    uv pip install -e .

    # Optional: install extra dependencies
    uv pip install -e ".[dev]"   # development
    uv pip install -e ".[docs]"  # documentation (Sphinx)

Quick Start
-----------

.. code-block:: python

    import rtnn

    # Version info
    print(f"RTnn version: {rtnn.__version__}")
    print(f"Author: {rtnn.__author__}")

    # Initialize model
    model = rtnn.RNN_LSTM(
        feature_channel=6,
        output_channel=4,
        hidden_size=256,
        num_layers=3,
    )

    # Prepare data
    # dataset = rtnn.DataPreprocessor(...)

    # Train or evaluate

Command Line Interface
----------------------

.. code-block:: bash

    # Show version
    rtnn --version

    # Show help
    rtnn --help

    # Train a model
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
        --embed_size "256" \  # Embedding size (just for transformer)
        --nhead "4" \  # Number of attention heads (just for transformer)
        --forward_expansion "4" \  # Feed-forward expansion factor (just for transformer)
        --dropout "0.1" \  # Dropout rate
        --model_name "lstm_h256_l3_d0d1" \  # Model identifier
        --batch_size "4" \  # Batch size
        --num_epochs "100" \  # Number of training epochs
        --learning_rate "0.0001" \  # Learning rate
        --loss_type "huber" \  # Loss function
        --beta "0.5" \  # Loss weighting parameter
        --beta_delta "1.0" \  # Secondary loss scaling factor
        --train_data_files "/path/to/training/data" \  # Training NetCDF4 dataset path
        --test_data_files "/path/to/testing/data" \  # Testing NetCDF4 dataset path
        --train_years "1995-1999" \  # Training time range
        --test_year "2000" \  # Test year
        --norm "log1p_standard" \  # Normalization method
        --num_workers "4" \  # DataLoader worker threads
        --save_model "True" \  # Save final model
        --save_checkpoint_name "model" \  # Checkpoint filename
        --save_per_samples "10000" \  # Save interval (samples)
        --run_type "train" \  # Run mode: training
        --seed "42" \  # Random seed
        --debug "False"  # Debug mode

    # Evaluate / inference a model
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

Project Structure
-----------------

::

    rtnn/
    ├── src/rtnn/                 # Main package
    │   ├── __init__.py
    │   ├── __main__.py           # CLI entry point
    │   ├── version.py            # Package version
    │   ├── dataset.py            # DataPreprocessor and dataset utilities
    │   ├── evaluater.py          # Metrics, loss, and evaluation logic
    │   ├── diagnostics.py        # Visualization and diagnostics tools
    │   ├── logger.py             # Logging utilities
    │   ├── main.py               # Training / inference pipeline
    │   ├── model_loader.py       # Model factory and loading utilities
    │   ├── model_utils.py        # Model helper functions
    │   ├── utils.py              # General utilities
    │   └── models/               # Neural network architectures
    │       ├── rnn.py            # LSTM / GRU models
    │       ├── fcn.py            # Fully connected networks
    │       ├── transformer.py    # Transformer encoder
    │       ├── mlp.py            # Multi-layer perceptron
    │
    ├── tests/                    # Unit tests
    │   ├── test_rnn.py
    │   ├── test_fcn.py
    │   ├── test_transformer.py
    │   ├── test_mlp.py
    │   ├── test_dataset.py
    │   ├── test_evaluater.py
    │   ├── test_model_loader.py
    │   ├── test_diagnostics.py
    │   ├── test_model_utils.py
    │   ├── test_utils.py
    │   └── test_runner.py
    │
    ├── docs/                    # Documentation (Sphinx)
    │   ├── source/
    │   └── build/
    │
    ├── .github/workflows/       # CI/CD pipelines
    │   ├── ci.yaml
    │   └── docs.yml
    │
    ├── pyproject.toml           # Package configuration
    ├── README.rst               # Project overview
    └── LICENSE                  # CC BY-NC-SA 4.0

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

RTnn uses Python's built-in unittest framework for testing.

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

References
----------

The development of RTnn has been informed by the following foundational works in neural network-based radiative transfer emulation:

.. code-block:: bibtex


    @article{chevallier1998neural,
        title={A Neural Network Approach for a Fast and Accurate Computation of a Longwave Radiative Budget},
        author={Chevallier, F. and Chéruy, F. and Scott, N. A. and Chédin, A.},
        journal={Journal of Applied Meteorology},
        volume={37},
        number={11},
        pages={1385--1397},
        year={1998}
    }

    @article{chevallier2000use,
        title={Use of a Neural‐network‐based Long‐wave Radiative‐transfer Scheme in the ECMWF Atmospheric Model},
        author={Chevallier, F. and Morcrette, J. ‐J. and Chéruy, F. and Scott, N. A.},
        journal={Quarterly Journal of the Royal Meteorological Society},
        volume={126},
        number={563},
        pages={761--776},
        year={2000}
    }


    @techreport{krasnopolsky2014nn,
        title={NN-TSV, NCEP Neural Network Training and Validation System: Brief Description of NN Background and Training Software},
        author={Krasnopolʹsky, V. M.},
        year={2014},
        doi={10.7289/V5QR4V2Z}
    }

    @article{krasnopolsky2010accurate,
        title={Accurate and Fast Neural Network Emulations of Model Radiation for the NCEP Coupled Climate Forecast System: Climate Simulations and Seasonal Predictions},
        author={Krasnopolsky, V. M. and Fox-Rabinovitz, M. S. and Hou, Y. T. and Lord, S. J. and Belochitski, A. A.},
        journal={Monthly Weather Review},
        volume={138},
        number={5},
        pages={1822--1842},
        year={2010}
    }

    @article{lagerquist2021using,
        title={Using Deep Learning to Emulate and Accelerate a Radiative-Transfer Model},
        author={Lagerquist, Ryan and Turner, David and Ebert-Uphoff, Imme and Stewart, Jebb and Hagerty, Venita},
        journal={Journal of Atmospheric and Oceanic Technology},
        year={2021}
    }

    @article{liang2022deep,
        title={A Deep-Learning-Based Microwave Radiative Transfer Emulator for Data Assimilation and Remote Sensing},
        author={Liang, Xingming and Garrett, Kevin and Liu, Quanhua and Maddy, Eric S. and Ide, Kayo and Boukabara, Sid},
        journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
        volume={15},
        pages={8819--8833},
        year={2022}
    }

    @article{liu2020radnet,
        title={RadNet 1.0: Exploring Deep Learning Architectures for Longwave Radiative Transfer},
        author={Liu, Ying and Caballero, Rodrigo and Monteiro, Joy Merwin},
        journal={Geoscientific Model Development},
        volume={13},
        number={9},
        pages={4399--4412},
        year={2020}
    }

    @article{mishra2021physics,
        title={Physics Informed Neural Networks for Simulating Radiative Transfer},
        author={Mishra, Siddhartha and Molinaro, Roberto},
        journal={Journal of Quantitative Spectroscopy and Radiative Transfer},
        volume={270},
        pages={107705},
        year={2021}
    }

    @article{mu2023radiative,
        title={A Radiative Transfer Deep Learning Model Coupled into WRF with a Generic Fortran Torch Adaptor},
        author={Mu, Bin and Chen, Lu and Yuan, Shijin and Qin, Bo},
        journal={Frontiers in Earth Science},
        volume={11},
        pages={1149566},
        year={2023}
    }

    @article{pal2019using,
        title={Using Deep Neural Networks as Cost‐Effective Surrogate Models for Super‐Parameterized E3SM Radiative Transfer},
        author={Pal, Anikesh and Mahajan, Salil and Norman, Matthew R.},
        journal={Geophysical Research Letters},
        volume={46},
        number={11},
        pages={6069--6079},
        year={2019}
    }

    @article{song2021improved,
        title={Improved Weather Forecasting Using Neural Network Emulation for Radiation Parameterization},
        author={Song, Hwan‐Jin and Roh, Soonyoung},
        journal={Journal of Advances in Modeling Earth Systems},
        volume={13},
        number={10},
        pages={e2021MS002609},
        year={2021}
    }

    @article{ukkonen2022exploring,
        title={Exploring Pathways to More Accurate Machine Learning Emulation of Atmospheric Radiative Transfer},
        author={Ukkonen, Peter},
        journal={Journal of Advances in Modeling Earth Systems},
        volume={14},
        number={4},
        pages={e2021MS002875},
        year={2022}
    }

    @article{yao2023physics,
        title={A Physics‐Incorporated Deep Learning Framework for Parameterization of Atmospheric Radiative Transfer},
        author={Yao, Yichen and Zhong, Xiaohui and Zheng, Yongjun and Wang, Zhibin},
        journal={Journal of Advances in Modeling Earth Systems},
        volume={15},
        number={5},
        pages={e2022MS003445},
        year={2023}
    }
