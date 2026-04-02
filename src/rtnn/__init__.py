"""RTnn: Radiative Transfer Neural Networks for Climate Science."""

__author__ = "Kazem Ardaneh"
__license__ = "Creative Commons Attribution-NonCommercial-ShareAlike 4.0"
__copyright__ = "2026, CNRS / IPSL / Sorbonne University"
__description__ = "Radiative Transfer Neural Networks for Climate Science"

# Import version info
from rtnn.version import __version__, __version_info__, get_version

# Import logger
from rtnn.logger import Logger

# Import main components for easy access
from rtnn.data_helper_lsm import DataPreprocessor
from rtnn.model_helper import ModelUtils
from rtnn.evaluate_helper import (
    check_accuracy_evaluate_lsm,
    get_loss_function,
    unnorm_mpas,
    calc_abs,
)
from rtnn.models.rnn import RNN_LSTM, RNN_GRU
from rtnn.models.fcn import FCN
from rtnn.models.Transformer import Encoder as TransformerEncoder
from rtnn.models.UNet1D import UNET
from rtnn.models.DimChangeModule import DimChange

__all__ = [
    "__version__",
    "__version_info__",
    "get_version",
    "__author__",
    "__license__",
    "__copyright__",
    "Logger",  # Add logger to exports
    "DataPreprocessor",
    "ModelUtils",
    "check_accuracy_evaluate_lsm",
    "get_loss_function",
    "unnorm_mpas",
    "calc_abs",
    "RNN_LSTM",
    "RNN_GRU",
    "FCN",
    "TransformerEncoder",
    "UNET",
    "DimChange",
]
