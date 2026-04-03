Project Structure
=================

::

   rtnn/
   ├── src/rtnn/
   │   ├── __init__.py
   │   ├── version.py
   │   ├── main.py          # Main training script
   │   ├── dataset.py   # Data preprocessing
   │   ├── evaluater.py   # Evaluation metrics
   │   ├── utils.py       # File utilities
   │   ├── model_utils.py      # Model utilities
   │   ├── model_loader.py     # Model factory
   │   ├── diagnostics.py       # Visualization
   │   ├── stats.py             # Statistics computation
   │   ├── logger.py            # Rich logging
   │   └── models/              # Model architectures
   │       ├── __init__.py
   │       ├── rnn.py           # LSTM/GRU
   │       ├── fcn.py           # Fully Connected Network
   │       ├── Transformer.py   # Transformer encoder
   │       └── DimChangeModule.py
   ├── tests/                   # Unit tests
   ├── docs/                    # Documentation
   ├── pyproject.toml
   └── README.rst
