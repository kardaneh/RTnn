Project Structure
=================

::

   rtnn/
   ├── src/rtnn/
   │   ├── __init__.py
   │   ├── version.py
   │   ├── main_lsm.py          # Main training script
   │   ├── data_helper_lsm.py   # Data preprocessing
   │   ├── evaluate_helper.py   # Evaluation metrics
   │   ├── file_helper.py       # File utilities
   │   ├── model_helper.py      # Model utilities
   │   ├── model_prepare.py     # Model factory
   │   ├── plot_helper.py       # Visualization
   │   ├── stats.py             # Statistics computation
   │   ├── logger.py            # Rich logging
   │   └── models/              # Model architectures
   │       ├── __init__.py
   │       ├── rnn.py           # LSTM/GRU
   │       ├── fcn.py           # Fully Connected Network
   │       ├── Transformer.py   # Transformer encoder
   │       ├── UNet1D.py        # 1D UNet
   │       └── DimChangeModule.py
   ├── tests/                   # Unit tests
   ├── docs/                    # Documentation
   ├── pyproject.toml
   └── README.rst
