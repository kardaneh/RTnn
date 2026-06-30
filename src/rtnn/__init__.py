# Copyright 2026 IPSL / CNRS / Sorbonne University
# Authors: Kazem Ardaneh
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

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
from rtnn.dataset import DataPreprocessor
from rtnn.model_utils import ModelUtils
from rtnn.evaluater import (
    run_validation_lsm,
    run_validation_cams,
    run_validation_reftrans,
    get_loss_function,
    unnorm_mpas,
    calc_abs,
)
from rtnn.models.rnn import RNN_LSTM, RNN_GRU
from rtnn.models.fcn import FCN
from rtnn.models.pinn import PINN
from rtnn.models.mlp import MLP, MLPResidual
from rtnn.models.transformer import EncoderTorch

__all__ = [
    "__version__",
    "__version_info__",
    "get_version",
    "__author__",
    "__license__",
    "__copyright__",
    "Logger",
    "DataPreprocessor",
    "ModelUtils",
    "run_validation_lsm",
    "run_validation_cams",
    "run_validation_reftrans",
    "get_loss_function",
    "unnorm_mpas",
    "calc_abs",
    "RNN_LSTM",
    "RNN_GRU",
    "FCN",
    "PINN",
    "MLP",
    "MLPResidual",
    "EncoderTorch",
]
