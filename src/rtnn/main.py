#!/usr/bin/env python3
# Copyright 2026 IPSL / CNRS / Sorbonne University
# Authors: Kazem Ardaneh
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

"""
RTnn (Radiative Transfer Neural Network) Training Pipeline

This module provides the main entry point for training neural network models
for radiative transfer calculations in climate modeling. It supports various
model architectures including LSTM, GRU, Transformer, and FCN.

The training pipeline includes:

- Data loading and preprocessing from NetCDF files
- Model initialization and configuration
- Training loop with progress tracking
- Validation and metric computation
- Checkpoint saving and model persistence
- Visualization and logging

Module Overview
---------------
This module implements a complete machine learning pipeline for radiative
transfer modeling in climate science. The pipeline is designed to handle
spatio-temporal data from NetCDF files, apply appropriate normalization,
train various neural network architectures, and evaluate model performance
using physics-informed metrics.

Key Features
------------
1. **Data Handling**

   - Automatic reading of NetCDF files with year-based filtering
   - Support for multiple normalization schemes (log1p_standard, standard, minmax)
   - Temporal batching and sequence generation
   - Multi-year training and held-out year testing

2. **Model Architectures**

   - LSTM (Long Short-Term Memory)
   - GRU (Gated Recurrent Unit)
   - Transformer with attention mechanism
   - FCN (Fully Connected Networks)
   - MLP with residual connections
   - Custom RT-specific architectures

3. **Training Features**

   - Configurable loss functions (MSE, MAE, NMSE, NMAE, SmoothL1, Huber)
   - Physics-informed weighted loss combining flux and absorption terms
   - Learning rate scheduling with ReduceLROnPlateau
   - Multi-GPU support with DataParallel
   - Checkpoint saving and resumption

4. **Evaluation Metrics**

   - NMAE (Normalized Mean Absolute Error)
   - NMSE (Normalized Mean Squared Error)
   - R² (Coefficient of Determination)
   - Conservation penalty for physical consistency

5. **Output and Visualization**

   - TensorBoard integration for real-time monitoring
   - Training/validation loss plots
   - Metric history visualization
   - Spatial-temporal density scatter plots
   - Automatic checkpoint saving (best, epoch, final)

Data Flow
---------
1. Parse command-line arguments (parse_args)
2. Setup directory structure and logging (setup_directories_and_logging)
3. Configure device and random seeds (setup_device_and_seed)
4. Load and preprocess data (get_data_files, create_datasets_and_loaders)
5. Compute normalization statistics (create_normalization_mapping)
6. Initialize model architecture (initialize_model)
7. Load checkpoint if resuming (load_checkpoint_if_requested)
8. Run inference or training loop
9. Generate plots and save results
"""

import os
import time
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import xarray as xr
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import random

# Import local modules
from rtnn.utils import FileUtils, EasyDict
from rtnn.dataset import DataPreprocessor
from rtnn.dataset_atm import RRTMGPDataPreprocessor
from rtnn.dataset_reftrans import REFTRANSDataPreprocessor
from rtnn.model_loader import load_model
from rtnn.model_utils import ModelUtils
from rtnn.evaluater import (
    unnorm_mpas,
    calc_abs,
    calc_heating_rates,
    run_validation_lsm,
    run_validation_cams,
    run_validation_reftrans,
    get_loss_function,
    MetricTracker,
    r2_all,
    nmae_all,
    nmse_all,
    mae_all,
    mse_all,
)
from rtnn.diagnostics import (
    plot_metric_histories,
    plot_loss_histories,
    plot_spatial_temporal_density,
    stats,
    stats_rrtmgp,
)
from rtnn.logger import Logger
from rtnn import __version__, __version_info__, __author__, __license__, __copyright__


def print_version():
    """Print detailed version information."""
    print(f"rtnn version {__version__}")
    print(f"Version info: {__version_info__}")
    print(f"Author: {__author__}")
    print(f"License: {__license__}")
    print(f"Copyright: {__copyright__}")


def parse_years(year_str):
    """
    Parse a year string into a list of integers.

    Supports hyphen-separated ranges (e.g., "1995-1999") and comma-separated
    lists (e.g., "1995,1997,1999"). Returns a list of integers.

    Parameters
    ----------
    year_str : str
        String containing years in range or comma-separated format.

    Returns
    -------
    list of int
        List of parsed years.

    Examples
    --------
    >>> parse_years("1995-1999")
    [1995, 1996, 1997, 1998, 1999]

    >>> parse_years("1995,1997,1999")
    [1995, 1997, 1999]
    """
    if "-" in year_str:
        start, end = map(int, year_str.split("-"))
        return list(range(start, end + 1))
    return list(map(int, year_str.split(",")))


def parse_args():
    """
    Parse command-line arguments for RTnn model training.

    Defines and parses command-line arguments required to configure and run the
    Radiative Transfer Neural Network (RTnn) training pipeline. This includes
    model architecture parameters, training hyperparameters, data configuration,
    and output settings.

    Returns
    -------
    argparse.Namespace
        Object containing parsed command-line arguments, grouped as follows:

        **Model architecture**
            type : str
                Model type (e.g., "lstm", "gru", "fcn", "fullyconnected",
                "transformer", "cnn", "mlp").
            hidden_size : int
                Size of hidden layers.
            num_layers : int
                Number of model layers.
            seq_length : int
                Length of input sequence.
            feature_channel : int
                Number of input feature channels.
            output_channel : int
                Number of output channels.
            embed_size : int
                Embedding dimension for transformer models.
            nhead : int
                Number of attention heads (transformer).
            forward_expansion : int
                Expansion factor for feed-forward layers.
            dropout : float
                Dropout rate.

        **Training hyperparameters**
            batch_size : int
                Number of samples per batch.
            tbatch : int
                Temporal batch length.
            num_epochs : int
                Number of training epochs.
            learning_rate : float
                Initial learning rate.
            loss_type : str
                Loss function (e.g., "mse", "mae", "nmae", "nmse", "wmse",
                "logcosh", "smoothl1", "huber").
            beta : float
                Weighting factor for loss components.
            beta_delta : float
                Delta parameter for Huber or SmoothL1 loss.
            num_workers : int
                Number of data loader workers.

        **Data configuration**
            train_data_files : str
                Path or pattern for training data files.
            test_data_files : str
                Path or pattern for testing data files.
            train_years : str
                Training years (comma-separated or range, e.g., "1995-1999").
            test_year : str
                Test year or range.
            norm : str
                Normalization scheme (e.g., "log1p_standard", "standard",
                "minmax", "none").
            dataset_type : str
                Dataset type (e.g., "ORC", "CAMS_RADSCHEME").

        **Output configuration**
            root_dir : str
                Root directory for all operations.
            main_folder : str
                Main output folder name.
            sub_folder : str
                Sub-folder name for the current run.
            prefix : str
                Prefix for saved files.
            model_name : str
                Custom model name (auto-generated if empty).
            save_model : bool
                Whether to save model checkpoints.
            save_checkpoint_name : str
                Base name for saved checkpoints.
            save_per_samples : int
                Save checkpoint every N samples.
            load_model : bool
                Whether to load an existing model.
            load_checkpoint_name : str
                Checkpoint file to load.
            inference : bool
                Run in inference-only mode.

    Examples
    --------
    >>> args = parse_args()
    >>> args.type
    'lstm'
    >>> args.batch_size
    16

    Command line usage
    ------------------
    $ rtnn --type lstm --hidden_size 128 --num_layers 3 --batch_size 32
    """
    parser = argparse.ArgumentParser(description="Train the RT model")

    parser.add_argument(
        "--version", action="store_true", help="Show version information and exit"
    )

    parser.add_argument(
        "--root_dir", type=str, default="./", help="Root directory for all operations"
    )

    parser.add_argument(
        "--train_data_files",
        type=str,
        default="./data/train/",
        help="Path to training dataset directory or file pattern",
    )

    parser.add_argument(
        "--test_data_files",
        type=str,
        default="./data/test/",
        help="Path to testing dataset directory or file pattern",
    )

    parser.add_argument(
        "--train_years",
        type=str,
        default="1995-1999",
        help="Training years as comma-separated list or range",
    )

    parser.add_argument(
        "--test_year", type=str, default="2000", help="Test year or year range"
    )

    parser.add_argument(
        "--main_folder", type=str, default="results", help="Main output folder name"
    )

    parser.add_argument(
        "--sub_folder",
        type=str,
        default="experiment",
        help="Sub-folder name for current run",
    )

    parser.add_argument(
        "--prefix", type=str, default="run", help="Prefix for saved files"
    )

    parser.add_argument(
        "--dataset_type",
        type=str,
        default="ORC",
        choices=["ORC", "CAMS_RADSCHEME", "CAMS_REFTRANS"],
        help="Type of dataset being processed",
    )

    parser.add_argument(
        "--norm",
        type=str,
        default="log1p_standard",
        choices=["log1p_standard", "standard", "minmax", "none"],
        help="Data normalization scheme",
    )

    parser.add_argument(
        "--loss_type",
        type=str,
        default="mse",
        choices=["mse", "mae", "nmae", "nmse", "smoothl1", "huber"],
        help="Loss function type",
    )

    parser.add_argument(
        "--beta_delta",
        type=float,
        default=1.0,
        help="Delta parameter for Huber/SmoothL1 loss",
    )

    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="Initial learning rate"
    )

    parser.add_argument(
        "--beta", type=float, default=0.05, help="Weighting factor for loss components"
    )

    parser.add_argument(
        "--batch_size", type=int, default=16, help="Number of samples per batch"
    )

    parser.add_argument(
        "--num_epochs", type=int, default=100, help="Total number of training epochs"
    )

    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of data loader workers"
    )

    parser.add_argument(
        "--type",
        type=str,
        default="lstm",
        choices=[
            "lstm",
            "gru",
            "fcn",
            "fullyconnected",
            "transformer",
            "encodertorch",
            "pinn",
            "mlp",
            "mlp_residual",
        ],
        help="Model architecture type",
    )

    parser.add_argument(
        "--hidden_size", type=int, default=256, help="Size of hidden layers"
    )

    parser.add_argument(
        "--num_layers", type=int, default=3, help="Number of model layers"
    )

    parser.add_argument(
        "--seq_length", type=int, default=10, help="Sequence length (None for default)"
    )

    parser.add_argument(
        "--feature_channel",
        type=int,
        default=6,
        help="Input feature channels (None for default)",
    )

    parser.add_argument(
        "--output_channel", type=int, default=4, help="Number of output channels"
    )

    parser.add_argument(
        "--embed_size",
        type=int,
        default=64,
        help="Embedding dimension (None for default)",
    )

    parser.add_argument(
        "--nhead", type=int, default=4, help="Attention heads (None for default)"
    )

    parser.add_argument(
        "--forward_expansion", type=int, default=4, help="Feed-forward expansion factor"
    )

    parser.add_argument(
        "--dropout", type=float, default=0.0, help="Dropout rate (None for no dropout)"
    )

    parser.add_argument(
        "--sblock_perc",
        type=float,
        default=0.5,
        help="Percentage of sites to use for training",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="",
        help="Custom model name (auto-generated if empty)",
    )

    parser.add_argument(
        "--save_model",
        type=lambda x: x.lower() == "true",
        default=False,
        help="Enable model checkpoint saving",
    )

    parser.add_argument(
        "--save_checkpoint_name",
        type=str,
        default="model",
        help="Base name for saved checkpoints",
    )

    parser.add_argument(
        "--save_per_samples",
        type=int,
        default=10000,
        help="Save checkpoint every N samples",
    )

    parser.add_argument(
        "--load_checkpoint_name",
        type=str,
        default="model.pth.tar",
        help="Checkpoint file to load",
    )

    parser.add_argument(
        "--debug",
        type=lambda x: x.lower() == "true",
        default=False,
        help="Enable or disable debug mode",
    )

    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    # Run configuration
    parser.add_argument(
        "--run_type",
        type=str,
        default="train",
        choices=["train", "resume_train", "inference"],
        help="Run type: 'train','resume_train', or 'inference' (mode determines whether to load checkpoint and how to handle training)",
    )

    args = parser.parse_args()
    return args


def setup_directories_and_logging(args):
    """
    Set up directory structure and logging infrastructure for experiments.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    paths : EasyDict
        Dictionary containing paths to created directories.
    logger : Logger
        Configured logger instance.
    """

    current_dir = os.path.abspath(__file__)
    parent_dir = os.path.dirname(current_dir)
    project_root = os.path.dirname(os.path.dirname(parent_dir))

    paths = EasyDict()
    paths.logs = os.path.join(project_root, "logs", args.main_folder, args.sub_folder)
    paths.results = os.path.join(
        project_root, "results", args.main_folder, args.sub_folder
    )
    paths.runs = os.path.join(project_root, "runs", args.main_folder, args.sub_folder)
    paths.checkpoints = os.path.join(
        project_root, "checkpoints", args.main_folder, args.sub_folder
    )
    paths.stats = os.path.join(project_root, "stats", args.main_folder, args.sub_folder)

    # Create directories
    for path in [paths.logs, paths.results, paths.runs, paths.checkpoints, paths.stats]:
        FileUtils.makedir(path)

    # Setup logger
    log_file = os.path.join(paths.logs, f"{args.prefix}_log.txt")
    logger = Logger(
        console_output=True, file_output=True, log_file=log_file, record=args.debug
    )
    logger.show_header("RTnn Training Pipeline")
    logger.success("Directories and logging infrastructure set up successfully.")

    return paths, logger


def log_configuration(args, paths, logger):
    """
    Log all configuration parameters to the provided logger.

    Parameters
    ----------
    args : argparse.Namespace
        Configuration object containing all experiment parameters.
    paths : EasyDict
        Dictionary containing paths to various experiment directories.
    logger : Logger
        Logger instance for outputting configuration information.
    """
    logger.info("=" * 60)
    logger.info("CONFIGURATION PARAMETERS")
    logger.info("=" * 60)

    # Execution mode
    logger.info("Execution Mode:")
    logger.info(f" |- Debug: {args.debug}")
    logger.info(f" |- Run type: {args.run_type}")
    logger.info(f" |- Save model: {args.save_model}")

    # Model architecture
    logger.info("\nModel Architecture:")
    logger.info(f" |- Type: {args.type}")
    logger.info(f" |- Hidden size: {args.hidden_size}")
    logger.info(f" |- Num layers: {args.num_layers}")
    logger.info(f" |- Seq length: {args.seq_length}")
    logger.info(f" |- Feature channels: {args.feature_channel}")
    logger.info(f" |- Output channels: {args.output_channel}")
    if args.type == "transformer":
        logger.info(f" |- Embed size: {args.embed_size}")
        logger.info(f" |- Attention heads: {args.nhead}")
        logger.info(f" |- Forward expansion: {args.forward_expansion}")
    logger.info(f" |- Dropout: {args.dropout}")

    # Training configuration
    logger.info("\nTraining Configuration:")
    logger.info(f" |- Number of epochs: {args.num_epochs}")
    logger.info(f" |- Batch size: {args.batch_size}")
    logger.info(f" |- Learning rate: {args.learning_rate}")
    logger.info(f" |- Number of workers: {args.num_workers}")
    logger.info(f" |- Random seed: {args.seed}")

    # Loss configuration
    logger.info("\nLoss Configuration:")
    logger.info(f" |- Loss type: {args.loss_type}")
    logger.info(f" |- Beta (absorption weight): {args.beta}")
    if args.loss_type in ["smoothl1", "huber"]:
        logger.info(f" |- Beta delta: {args.beta_delta}")

    # Data configuration
    logger.info("\nData Configuration:")
    logger.info(f" |- Train years: {args.train_years}")
    logger.info(f" |- Test year: {args.test_year}")
    logger.info(f" |- Train data path: {args.train_data_files}")
    logger.info(f" |- Test data path: {args.test_data_files}")
    logger.info(f" |- Dataset type: {args.dataset_type}")
    logger.info(f" |- Normalization: {args.norm}")

    # Output configuration
    logger.info("\nOutput Configuration:")
    logger.info(f" |- Main folder: {args.main_folder}")
    logger.info(f" |- Sub folder: {args.sub_folder}")
    logger.info(f" |- Prefix: {args.prefix}")

    # Checkpoint configuration
    logger.info("\nCheckpoint Configuration:")
    logger.info(f" |- Save model: {args.save_model}")
    logger.info(f" |- Save checkpoint name: {args.save_checkpoint_name}")
    logger.info(f" |- Save per samples: {args.save_per_samples}")
    logger.info(f" |- Load checkpoint name: {args.load_checkpoint_name}")

    # Directory paths
    logger.info("\nGenerated Directory Paths:")
    logger.info(f" |- Logs: {paths.logs}")
    logger.info(f" |- Results: {paths.results}")
    logger.info(f" |- TensorBoard: {paths.runs}")
    logger.info(f" |- Checkpoints: {paths.checkpoints}")
    logger.info(f" |- Statistics: {paths.stats}")

    logger.info("=" * 60)


def setup_device_and_seed(args, logger):
    """
    Set up device (GPU/CPU) and random seeds for reproducibility.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.
    logger : Logger
        Logger instance.

    Returns
    -------
    torch.device
        Device to use for computations.
    """

    logger.start_task("Setting up device and random seeds")
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    logger.info(f"Random seed set to: {args.seed}")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA version: {torch.version.cuda}")

    # Enable anomaly detection for debugging
    torch.autograd.set_detect_anomaly(True)
    logger.warning("Anomaly detection enabled (may slow down training)")

    logger.success("Device and random seeds set up successfully")
    return device


def get_data_files_lsm(args, logger):
    """
    Get training and testing data files based on year specifications.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.
    logger : Logger
        Logger instance.

    Returns
    -------
    tuple
        (train_files, test_files) lists of file paths.
    """

    logger.start_task("Getting data files")
    train_years = parse_years(args.train_years)
    test_years = parse_years(args.test_year)

    # Get training files
    train_files = []
    for year in train_years:
        pattern = f"{args.train_data_files}/rtnetcdf_*_{year}.nc"
        files = glob.glob(pattern)
        if not files:
            logger.warning(
                f"No training files found for year {year} with pattern {pattern}"
            )
        train_files.extend(files)
    train_files = sorted(train_files)

    # Get test files
    test_files = []
    for year in test_years:
        pattern = f"{args.test_data_files}/rtnetcdf_*_{year}.nc"
        files = glob.glob(pattern)
        if not files:
            logger.warning(
                f"No test files found for year {year} with pattern {pattern}"
            )
        test_files.extend(files)
    test_files = sorted(test_files)

    logger.info(f"Found {len(train_files)} training files:")
    for f in train_files[:5]:
        logger.info(f"  {f}")
    if len(train_files) > 5:
        logger.info(f"  ... and {len(train_files) - 5} more")

    logger.info(f"Found {len(test_files)} test files:")
    for f in test_files[:5]:
        logger.info(f"  {f}")
    if len(test_files) > 5:
        logger.info(f"  ... and {len(test_files) - 5} more")

    logger.success("Data files retrieved successfully")

    return train_files, test_files


def get_data_files_rrtmgp(args, logger=None):
    """
    Get training and testing data files for RRTMGP data.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments. Expected to contain:

        train_data_files : str
            Path to training data file

        test_data_files : str
            Path to test/validation data file

    logger : Logger
        Logger instance.

    Returns
    -------
    tuple
        (train_file, test_file) lists of file paths.
    """
    if logger is not None:
        logger.start_task("Getting RRTMGP data files")

    # Training file
    train_file = args.train_data_files
    if logger is not None:
        logger.info(f"Training file: {os.path.basename(train_file)}")

    # Test/Validation file
    test_file = args.test_data_files
    if logger is not None:
        logger.info(f"Test file: {os.path.basename(test_file)}")
        logger.success("Data files retrieved successfully")

    return train_file, test_file


def create_normalization_mapping_lsm(train_files, paths, logger):
    """
    Create normalization mapping from training data.

    Parameters
    ----------
    train_files : list
        List of training file paths.
    paths : EasyDict
        Directory paths.
    logger : Logger
        Logger instance.

    Returns
    -------
    dict
        Normalization statistics for each variable.
    """
    if not train_files:
        raise ValueError("No training files found for normalization statistics")

    logger.start_task(
        "Computing statistics for normalization",
        f"Using first training file: {train_files[0]}",
    )

    norm_mapping = stats([train_files[0]], logger, paths.stats, norm_mapping=None)

    for var, stat in norm_mapping.items():
        logger.info(f"Variable: {var}")
        for stat_name, value in stat.items():
            logger.info(f"  |- {stat_name}: {value}")

    logger.success(
        f"Normalization statistics computed for {len(norm_mapping)} variables"
    )

    return norm_mapping


def create_normalization_mapping_rrtmgp(
    file_path, output_dir=None, logger=None, norm_mapping=None, plots=False
):
    if norm_mapping is None:
        norm_mapping = {}

    if logger is not None:
        logger.info("Starting statistics computation for normalization")
        logger.info(f"Loading file: {file_path}")
    else:
        print(f"Loading file: {file_path}")

    ds = xr.open_dataset(file_path)
    # Gas features
    gas_input = ds["rrtmgp_sw_input"].values
    gas_reshaped = gas_input.reshape(-1, ds.sizes["layer"], ds.sizes["feature"])
    gas_names = ["tlay", "play", "h2o", "o3", "co2", "n2o", "ch4"]

    for i, name in enumerate(gas_names):
        data = gas_reshaped[..., i].flatten()
        norm_mapping[name] = stats_rrtmgp(data, name, output_dir, plots)

    # Cloud variables
    for name in ["cloud_lwp", "cloud_iwp", "cloud_fraction"]:
        data = ds[name].values.flatten()
        norm_mapping[name] = stats_rrtmgp(data, name, output_dir, plots)

    # Auxiliary variables
    data = ds["mu0"].values.flatten()
    norm_mapping["mu0"] = stats_rrtmgp(data, "mu0", output_dir, plots)
    data = ds["sfc_alb"].values.flatten()
    norm_mapping["sfc_alb"] = stats_rrtmgp(data, "sfc_alb", output_dir, plots)

    # Flux variables
    for name in ["rsd", "rsu", "rsd_dir", "toa_flux"]:
        data = ds[name].values.flatten()
        norm_mapping[name] = stats_rrtmgp(data, name, output_dir, plots)

    ds.close()
    if logger is not None:
        logger.info("Statistics computation completed")
    else:
        print("Statistics computation completed")
    return norm_mapping


def create_normalization_mapping_reftrans(
    file_path,
    output_dir,
    logger=None,
    norm_mapping=None,
    plots=False,
    sample_percentage=0.1,
):
    """
    Create normalization mapping for REFTRANS data using a random subset of sites.

    Parameters
    ----------
    file_path : str
        Path to NetCDF file
    output_dir : str
        Directory to save plots
    logger : Logger, optional
        Logger instance
    norm_mapping : dict, optional
        Dictionary to update with statistics
    plots : bool, optional
        Whether to generate histogram plots
    sample_percentage : float, optional
        Percentage of sites to use for computing statistics (default: 0.1 = 10%)

    Returns
    -------
    dict
        Normalization mapping dictionary
    """
    if norm_mapping is None:
        norm_mapping = {}

    if logger is not None:
        logger.info("Starting statistics computation for REFTRANS normalization")
        logger.info(f"Loading file: {file_path}")
        logger.info(f"Using {sample_percentage*100:.1f}% of sites for statistics")
    else:
        print(f"Loading file: {file_path}")
        print(f"Using {sample_percentage*100:.1f}% of sites for statistics")

    ds = xr.open_dataset(file_path)

    # Get dimensions
    n_layer = ds.sizes["layer"]
    n_gpt = ds.sizes["gpt"]
    n_site = ds.sizes["site"]

    # Randomly select sites
    n_sites_used = max(1, int(n_site * sample_percentage))
    selected_sites = sorted(random.sample(range(n_site), n_sites_used))

    if logger is not None:
        logger.info(
            f"Selected {len(selected_sites)} sites out of {n_site} for norm mapping"
        )
    else:
        print(f"Selected {len(selected_sites)} sites out of {n_site} for norm mapping")

    # Input features: tau, ssa, g, mu0 - using only selected sites
    input_names = ["tau_sw", "ssa_sw", "g_sw", "mu0"]
    for name in input_names:
        if name in ds.variables:
            # Extract only selected sites
            data = (
                ds[name].values[0, selected_sites, :, :].flatten()
                if "layer" in ds[name].dims
                else ds[name].values[0, selected_sites].flatten()
            )
            norm_mapping[name] = stats_rrtmgp(data, name, output_dir, plots)
            del data

    # Compute tnoscat using only selected sites
    if logger is not None:
        logger.info("Computing tnoscat statistics...")
    else:
        print("Computing tnoscat statistics...")

    # Get tau: (expt, site, layer, gpt) -> only selected sites
    tau = ds["tau_sw"].values[0, selected_sites, :, :]  # (n_sites_used, 60, 224)

    # Get mu0: (expt, site) -> only selected sites
    mu0 = ds["mu0"].values[0, selected_sites]  # (n_sites_used,)

    # Expand mu0 to match tau shape: (n_sites_used, 60, 224)
    mu0_expanded = np.tile(mu0[:, np.newaxis, np.newaxis], (1, n_layer, n_gpt))

    # Safe division
    mu0_safe = np.where(mu0_expanded > 1e-8, mu0_expanded, 1e-8)

    # Compute tnoscat: exp(-tau / mu0)
    tnoscat = np.exp(-tau / mu0_safe)

    # Flatten and compute statistics
    tnoscat_data = tnoscat.flatten()
    norm_mapping["tnoscat"] = stats_rrtmgp(tnoscat_data, "tnoscat", output_dir, plots)

    del tnoscat_data, tnoscat, mu0_expanded, mu0_safe

    # Output targets - using only selected sites
    output_names = ["rdif", "tdif", "rdir", "tdir"]
    for name in output_names:
        if name in ds.variables:
            data = ds[name].values[0, selected_sites, :, :].flatten()
            norm_mapping[name] = stats_rrtmgp(data, name, output_dir, plots)
            del data

    ds.close()

    if logger is not None:
        logger.info("Statistics computation completed")
    else:
        print("Statistics computation completed")

    return norm_mapping


def create_datasets_and_loaders_lsm(
    args, train_files, test_files, norm_mapping, logger
):
    """
    Create datasets and data loaders for training and validation.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.
    train_files : list
        Training file paths.
    test_files : list
        Test file paths.
    norm_mapping : dict
        Normalization statistics.
    logger : Logger
        Logger instance.

    Returns
    -------
    tuple
        (train_loader, test_loader, train_dataset, test_dataset)
    """

    # Define normalization types for each variable
    normalization_type = {
        "coszang": args.norm,
        "laieff_collim": args.norm,
        "laieff_isotrop": args.norm,
        "leaf_ssa": args.norm,
        "leaf_psd": args.norm,
        "rs_surface_emu": args.norm,
        "collim_alb": args.norm,
        "collim_tran": args.norm,
        "isotrop_alb": args.norm,
        "isotrop_tran": args.norm,
    }

    for var, norm in normalization_type.items():
        logger.info(f"Normalization for '{var}': '{norm}'")

    # Create training dataset
    logger.start_task("Creating training dataset", f"Files: {len(train_files)}")
    train_dataset = DataPreprocessor(
        logger=logger,
        dfs=train_files,
        stime=0,
        tbatch=1,
        training=True,
        norm_mapping=norm_mapping,
        normalization_type=normalization_type,
        debug=args.debug,
    )
    logger.success("Training dataset created successfully.")

    # Create test dataset
    test_tbatch = 4 if args.run_type == "inference" else 48
    logger.start_task("Creating test dataset", f"Files: {len(test_files)}")
    test_dataset = DataPreprocessor(
        logger=logger,
        dfs=test_files,
        stime=0,
        tbatch=test_tbatch,
        training=False,
        norm_mapping=norm_mapping,
        normalization_type=normalization_type,
        debug=args.debug,
    )
    logger.success("Test dataset created successfully.")

    logger.start_task("Creating data loaders")
    # Create data loaders
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,  # Use single worker for test loader to avoid potential issues
        pin_memory=True,
    )

    logger.info(f"Training dataset size: {len(train_dataset)} samples")
    logger.info(f"Test dataset size: {len(test_dataset)} samples")
    logger.info(f"Training batches per epoch: {len(train_loader)}")
    logger.info(f"Test batches: {len(test_loader)}")
    logger.success("Data loaders created successfully.")

    return train_loader, test_loader, normalization_type, train_dataset, test_dataset


def create_datasets_and_loaders_rrtmgp(
    args, train_file, test_file, norm_mapping, logger=None, normalization_type=None
):
    """
    Create datasets and data loaders for RRTMGP training and validation.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments. Expected to contain:
        - batch_size : int
        - num_workers : int
        - sblock_perc : float, optional (default: 0.6)
        - debug : bool, optional (default: False)
    train_file : list
        Training file path (should be a single file).
    test_file : list
        Test file path (should be a single file).
    norm_mapping : dict
        Normalization statistics from stats_rrtmgp function.
    logger : Logger, optional
        Logger instance. If None, no logging will be performed.
    normalization_type : dict, optional
        Normalization type for each variable. If None, defaults will be used.

    Returns
    -------
    tuple
        (train_loader, test_loader, normalization_type, train_dataset, test_dataset)
    """

    # Default normalization types for RRTMGP data
    if normalization_type is None:
        normalization_type = {
            # Gas features
            "tlay": "standard",
            "play": "log1p_standard",
            "h2o": "log1p_standard",
            "o3": "log1p_standard",
            "co2": "log1p_standard",
            "n2o": "log1p_standard",
            "ch4": "log1p_standard",
            # Cloud properties
            "cloud_lwp": "standard",
            "cloud_iwp": "standard",
            # Auxiliaries
            "mu0": "minmax",
            "sfc_alb": "minmax",
            # Flux targets
            "rsd": "minmax",
            "rsu": "minmax",
        }

    if logger is not None:
        logger.start_task("Creating datasets and loaders for RRTMGP data")
        logger.info("Normalization types:")
        for var, norm in normalization_type.items():
            logger.info(f"  {var}: '{norm}'")

    # Create training dataset
    if logger is not None:
        logger.start_task("Creating training dataset", f"File: {train_file}")

    train_dataset = RRTMGPDataPreprocessor(
        logger=logger,
        path=train_file,
        training=True,
        norm_mapping=norm_mapping,
        normalization_type=normalization_type,
        debug=args.debug,
        sblock_perc=args.sblock_perc,
    )

    if logger is not None:
        logger.success("Training dataset created successfully.")

    # Create test dataset
    if logger is not None:
        logger.start_task("Creating test dataset", f"File: {test_file}")

    test_dataset = RRTMGPDataPreprocessor(
        logger=logger,
        path=test_file,
        training=False,  # Use all sites
        norm_mapping=norm_mapping,
        normalization_type=normalization_type,
        debug=args.debug,
        sblock_perc=1.0,  # 100% of sites
    )

    if logger is not None:
        logger.success("Test dataset created successfully.")

    # Create data loaders
    if logger is not None:
        logger.start_task("Creating data loaders")

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    if logger is not None:
        logger.info(f"Training dataset size: {len(train_dataset)} experiments")
        logger.info(f"Test dataset size: {len(test_dataset)} experiments")
        logger.info(f"Training batches per epoch: {len(train_loader)}")
        logger.info(f"Test batches: {len(test_loader)}")
        logger.success("Data loaders created successfully.")

    return train_loader, test_loader, normalization_type, train_dataset, test_dataset


def create_datasets_and_loaders_reftrans(
    args, train_file, test_file, norm_mapping, logger=None, normalization_type=None
):
    """
    Create datasets and data loaders for REFTRANS training and validation.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.
    train_file : str
        Training file path.
    test_file : str
        Test file path.
    norm_mapping : dict
        Normalization statistics.
    logger : Logger, optional
        Logger instance.
    normalization_type : dict, optional
        Normalization type for each variable.

    Returns
    -------
    tuple
        (train_loader, test_loader, normalization_type, train_dataset, test_dataset)
    """

    # Default normalization types for REFTRANS data
    if normalization_type is None:
        normalization_type = {
            # Input features (5 features)
            "tau_sw": "log1p_standard",  # tau is positive and skewed
            "ssa_sw": "log1p_standard",  # ssa is in [0,1]
            "g_sw": "log1p_standard",  # g is in [0,0.5]
            "mu0": "log1p_standard",  # mu0 is in [0,1]
            "tnoscat": "minmax",  # tnoscat is in [0,1]
            # Output targets
            "rdif": "log1p_standard",  # reflectance in [0,1]
            "tdif": "log1p_standard",  # transmittance in [0,1]
            "rdir": "log1p_standard",  # reflectance in [0,1]
            "tdir": "log1p_standard",  # transmittance in [0,1]
        }

    if logger is not None:
        logger.start_task("Creating datasets and loaders for REFTRANS data")
        logger.info("Normalization types:")
        for var, norm in normalization_type.items():
            logger.info(f"  {var}: '{norm}'")

    # Create training dataset
    if logger is not None:
        logger.start_task("Creating training dataset", f"File: {train_file}")

    train_dataset = REFTRANSDataPreprocessor(
        logger=logger,
        path=train_file,
        training=True,
        norm_mapping=norm_mapping,
        normalization_type=normalization_type,
        debug=args.debug,
    )

    if logger is not None:
        logger.success("Training dataset created successfully.")

    # Create test dataset
    if logger is not None:
        logger.start_task("Creating test dataset", f"File: {test_file}")

    test_dataset = REFTRANSDataPreprocessor(
        logger=logger,
        path=test_file,
        training=False,
        norm_mapping=norm_mapping,
        normalization_type=normalization_type,
        debug=args.debug,
    )

    if logger is not None:
        logger.success("Test dataset created successfully.")

    # Create data loaders
    if logger is not None:
        logger.start_task("Creating data loaders")

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    if logger is not None:
        logger.info(f"Training dataset size: {len(train_dataset)} experiments")
        logger.info(f"Test dataset size: {len(test_dataset)} experiments")
        logger.info(f"Training batches per epoch: {len(train_loader)}")
        logger.info(f"Test batches: {len(test_loader)}")
        logger.success("Data loaders created successfully.")

    return train_loader, test_loader, normalization_type, train_dataset, test_dataset


def initialize_model(args, device, logger):
    """
    Initialize the model architecture.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.
    device : torch.device
        Device to place model on.
    logger : Logger
        Logger instance.

    Returns
    -------
    torch.nn.Module
        Initialized model.
    """
    logger.start_task("Initializing model", f"Type: {args.type}")

    model = load_model(args)
    _ = ModelUtils.get_parameter_number(model, logger)

    model = model.to(device)

    # Enable DataParallel for multi-GPU training
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)

    logger.success("Model initialized successfully.")
    return model


def load_checkpoint_if_requested(args, model, optimizer, paths, device, logger):
    """
    Load model checkpoint if requested using ModelUtils.load_training_checkpoint().

    This function leverages ModelUtils.load_training_checkpoint() which handles:
    - DataParallel compatibility
    - Loading model and optimizer states
    - Extracting training state (epoch, samples, metrics, etc.)

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.
    model : torch.nn.Module
        Model to load weights into.
    optimizer : torch.optim.Optimizer
        Optimizer to restore state.
    paths : EasyDict
        Directory paths.
    device : torch.device
        Device for loading.
    logger : Logger
        Logger instance.

    Returns
    -------
    tuple
        (start_epoch, samples_processed, batches_processed, best_val_loss,
         best_epoch, checkpoint, train_loss_history, valid_loss_history,
         valid_metrics_history)
    """
    # Only load if requested
    if args.run_type == "train":
        logger.info("Starting new training run (no checkpoint loading)")
        return 0, 0, 0, float("inf"), 0, None, [], [], {}
    else:
        logger.info(f"Run type: {args.run_type} - checkpoint loading will be attempted")

    checkpoint_path = os.path.join(paths.checkpoints, args.load_checkpoint_name)

    # Debug information
    if args.debug:
        logger.info("=" * 60)
        logger.info("CHECKPOINT LOADING DEBUG INFO")
        logger.info("=" * 60)
        logger.info(f"Checkpoint path: {checkpoint_path}")
        logger.info(f"Model checkpoint directory: {paths.checkpoints}")
        logger.info(f"Load checkpoint name: {args.load_checkpoint_name}")
        logger.info(f"Full checkpoint path exists: {os.path.exists(checkpoint_path)}")

    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint not found at: {checkpoint_path}")
        logger.error("Cannot load model without checkpoint. Exiting.")
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load checkpoint using ModelUtils
    if args.debug:
        logger.info(f"Loading checkpoint from: {checkpoint_path}")

    (
        epoch,
        samples_processed,
        batches_processed,
        best_val_loss,
        best_epoch,
        checkpoint,
    ) = ModelUtils.load_training_checkpoint(
        checkpoint_path, model, optimizer, device, logger=logger
    )

    # Extract history from checkpoint if available
    train_loss_history = checkpoint.get("train_loss_history", [])
    valid_loss_history = checkpoint.get("valid_loss_history", [])
    valid_metrics_history = checkpoint.get("valid_metrics_history", {})

    # Debug logging
    if args.debug and checkpoint is not None:
        logger.info("Checkpoint loaded successfully")
        logger.info(f" |- Epoch: {epoch}")
        logger.info(f" |- Samples processed: {samples_processed:,}")
        logger.info(f" |- Batches processed: {batches_processed:,}")
        logger.info(f" |- Best validation loss: {best_val_loss:.6f}")
        logger.info(f" |- Best epoch: {best_epoch}")
        logger.info("Checkpoint keys available:")
        for key in checkpoint.keys():
            if isinstance(checkpoint[key], (list, dict)):
                if key in ["train_loss_history", "valid_loss_history"]:
                    logger.info(f" |- {key}: list with {len(checkpoint[key])} elements")
                elif key == "valid_metrics_history":
                    logger.info(f" |- {key}: dict with {len(checkpoint[key])} keys")
                elif key == "args":
                    logger.info(
                        f" |- {key}: dict with {len(checkpoint[key])} arguments"
                    )
                else:
                    logger.info(f" |- {key}: {type(checkpoint[key]).__name__}")
            else:
                logger.info(f" |- {key}: {checkpoint[key]}")

    # Determine starting epoch (resume training or continue from checkpoint)
    start_epoch = epoch + 1 if args.run_type == "resume_train" else epoch

    if args.run_type == "resume_train":
        logger.info(f"Resuming training from epoch {start_epoch}")
        if args.debug:
            logger.info(f"Current train_loss_history length: {len(train_loss_history)}")
            logger.info(f"Current valid_loss_history length: {len(valid_loss_history)}")
    else:
        logger.info(f"Model loaded for 'inference' as {args.run_type} == 'inference'")

    return (
        start_epoch,
        samples_processed,
        batches_processed,
        best_val_loss,
        best_epoch,
        checkpoint,
        train_loss_history,
        valid_loss_history,
        valid_metrics_history,
    )


def train_epoch_lsm(
    model,
    train_loader,
    optimizer,
    loss_func,
    metric_funcs,
    metric_names,
    output_keys,
    train_metrics,
    train_loss_tracker,
    norm_mapping,
    normalization_type,
    index_mapping,
    device,
    args,
    epoch,
    writer,
    global_step,
    logger,
    n_pft=15,
    n_bands=2,
    n_chans=4,
):
    """
    Train for one epoch.

    Returns
    -------
    tuple
        (average_train_loss, updated_global_step)
    """

    model.train()

    # Reset metrics
    for meter in train_metrics.values():
        meter.reset()
    train_loss_tracker.reset()

    loop = tqdm(
        enumerate(train_loader),
        total=len(train_loader),
        desc=f"Epoch {epoch}",
        leave=False,
    )

    for batch_idx, (feature, targets) in loop:
        if epoch == 0 and batch_idx == 0:
            logger.info("First batch shapes:")
            logger.info(f"  Feature shape: {feature.shape}")
            logger.info(f"  Targets shape: {targets.shape}")

        # Reshape inputs
        feature_shape = feature.shape
        target_shape = targets.shape
        inner_batch_size = feature_shape[0] * feature_shape[1]

        feature = feature.reshape(
            inner_batch_size, feature_shape[2], feature_shape[3]
        ).to(device=device)
        targets = targets.reshape(
            inner_batch_size, target_shape[2], target_shape[3]
        ).to(device=device)

        # Forward pass
        predicts = model(feature)

        # Reshape to (batch, 4, n_pft, n_bands, seq)
        pred_reshaped = predicts.reshape(
            inner_batch_size, n_chans, n_pft, n_bands, target_shape[3]
        )
        targ_reshaped = targets.reshape(
            inner_batch_size, n_chans, n_pft, n_bands, target_shape[3]
        )

        predicts_unnorm, targets_unnorm = unnorm_mpas(
            pred_reshaped,
            targ_reshaped,
            norm_mapping,
            normalization_type,
            index_mapping,
        )
        assert (
            predicts_unnorm.shape == pred_reshaped.shape
        ), f"Expected predicts_unnorm shape {pred_reshaped.shape}, got {predicts_unnorm.shape}"

        assert (
            targets_unnorm.shape == targ_reshaped.shape
        ), f"Expected targets_unnorm shape {targ_reshaped.shape}, got {targets_unnorm.shape}"

        # Calculate absorption
        (
            abs12_predict,
            abs12_target,
            abs34_predict,
            abs34_target,
            conservation_penalty,
        ) = calc_abs(predicts_unnorm, targets_unnorm)
        if epoch == 0 and batch_idx == 0:
            logger.info(f"conservation_penalty: {conservation_penalty}")

        expected_abs_shape = (inner_batch_size, 1, n_pft, n_bands, target_shape[3] - 1)
        assert (
            abs12_predict.shape == expected_abs_shape
        ), f"Expected abs12_predict shape {expected_abs_shape}, got {abs12_predict.shape}"
        assert (
            abs12_target.shape == expected_abs_shape
        ), f"Expected abs12_target shape {expected_abs_shape}, got {abs12_target.shape}"
        assert (
            abs34_predict.shape == expected_abs_shape
        ), f"Expected abs34_predict shape {expected_abs_shape}, got {abs34_predict.shape}"
        assert (
            abs34_target.shape == expected_abs_shape
        ), f"Expected abs34_target shape {expected_abs_shape}, got {abs34_target.shape}"

        # Compute metrics
        output_dict = {
            "fluxes": (predicts, targets),
            "abs12": (abs12_predict, abs12_target),
            "abs34": (abs34_predict, abs34_target),
        }

        for key in output_keys:
            pred, tgt = output_dict[key]
            for metric in metric_names:
                metric_key = f"{key}_{metric}"
                if metric_key not in train_metrics:
                    logger.error(
                        f"Metric key '{metric_key}' not found in train_metrics"
                    )
                    raise KeyError(
                        f"Metric key '{metric_key}' not found in train_metrics"
                    )
                count, value = metric_funcs[metric](pred, tgt)
                train_metrics[metric_key].update(value.item(), count)

        # Compute loss
        main_count, main_val = predicts.numel(), loss_func(predicts, targets)

        abs12_count, abs12_val = (
            abs12_predict.numel(),
            loss_func(abs12_predict, abs12_target),
        )
        abs34_count, abs34_val = (
            abs34_predict.numel(),
            loss_func(abs34_predict, abs34_target),
        )

        weighted_loss = (1.0 - args.beta) * main_val * main_count + args.beta * (
            abs12_val * abs12_count + abs34_val * abs34_count
        )

        total_count = (1.0 - args.beta) * main_count + args.beta * (
            abs12_count + abs34_count
        )

        total_loss = weighted_loss / total_count

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Update trackers
        train_loss_tracker.update(total_loss.item(), 1)

        # Update progress bar
        loop.set_postfix(loss=total_loss.item())

        # TensorBoard logging
        writer.add_scalar(
            "train/total_loss", total_loss.item(), global_step=global_step
        )
        global_step += args.batch_size

    return train_loss_tracker.getmean(), global_step


def train_epoch_reftrans(
    model,
    train_loader,
    optimizer,
    loss_func,
    metric_funcs,
    metric_names,
    output_keys,
    train_metrics,
    train_loss_tracker,
    norm_mapping,
    normalization_type,
    index_mapping,
    device,
    args,
    epoch,
    writer,
    global_step,
    logger,
):
    """
    Train for one epoch on REFTRANS data.

    Data format: (B, C, T) where C=4
    Channels: [rdif, tdif, rdir, tdir]
    - abs12: diffuse absorption (channels 0,1)
    - abs34: direct absorption (channels 2,3)

    Returns
    -------
    tuple
        (average_train_loss, updated_global_step)
    """

    model.train()

    # Reset metrics
    for meter in train_metrics.values():
        meter.reset()
    train_loss_tracker.reset()

    loop = tqdm(
        enumerate(train_loader),
        total=len(train_loader),
        desc=f"Epoch {epoch}",
        leave=False,
    )

    for batch_idx, (features, targets) in loop:
        if epoch == 0 and batch_idx == 0:
            logger.info("First batch shapes (REFTRANS):")
            logger.info(
                f"  Feature shape: {features.shape}"
            )  # (batch, n_samples, n_features, n_layer)
            logger.info(
                f"  Targets shape: {targets.shape}"
            )  # (batch, n_samples, 4, n_layer)

        # Reshape inputs: flatten batch and sample dimensions
        feature_shape = features.shape
        target_shape = targets.shape
        inner_batch_size = feature_shape[0] * feature_shape[1]  # batch * n_samples

        # Reshape to (inner_batch, n_features, n_layer)
        features = features.reshape(
            inner_batch_size, feature_shape[2], feature_shape[3]
        ).to(device=device)

        # Reshape to (inner_batch, n_outputs, n_layer)
        targets = targets.reshape(
            inner_batch_size, target_shape[2], target_shape[3]
        ).to(device=device)

        # Forward pass
        # Model expects: (inner_batch, n_features, n_layer)
        # Model returns: (inner_batch, n_outputs, n_layer)
        predicts = model(features)  # (inner_batch, 4, n_layer)

        # Denormalize predictions and targets using unnorm_mpas
        # Since data is in (B, C, T) format with C=4
        predicts_unnorm, targets_unnorm = unnorm_mpas(
            predicts,
            targets,
            norm_mapping,
            normalization_type,
            index_mapping,
        )

        assert (
            predicts_unnorm.shape == predicts.shape
        ), f"Expected predicts_unnorm shape {predicts.shape}, got {predicts_unnorm.shape}"
        assert (
            targets_unnorm.shape == targets.shape
        ), f"Expected targets_unnorm shape {targets.shape}, got {targets_unnorm.shape}"

        # Calculate absorption using existing calc_abs
        # channels 0,1 -> abs12 (diffuse: rdif, tdif)
        # channels 2,3 -> abs34 (direct: rdir, tdir)
        (
            abs12_predict,
            abs12_target,
            abs34_predict,
            abs34_target,
            conservation_penalty,
        ) = calc_abs(predicts_unnorm, targets_unnorm)

        if epoch == 0 and batch_idx == 0:
            logger.info(f"Conservation penalty: {conservation_penalty.item():.6f}")
            logger.info(f"  abs12_predict shape: {abs12_predict.shape}")
            logger.info(f"  abs12_target shape: {abs12_target.shape}")
            logger.info(f"  abs34_predict shape: {abs34_predict.shape}")
            logger.info(f"  abs34_target shape: {abs34_target.shape}")

        # Compute metrics
        output_dict = {
            "fluxes": (predicts, targets),
            "abs12": (abs12_predict, abs12_target),
            "abs34": (abs34_predict, abs34_target),
        }

        for key in output_keys:
            pred, tgt = output_dict[key]
            for metric in metric_names:
                metric_key = f"{key}_{metric}"
                if metric_key not in train_metrics:
                    logger.error(
                        f"Metric key '{metric_key}' not found in train_metrics"
                    )
                    raise KeyError(
                        f"Metric key '{metric_key}' not found in train_metrics"
                    )
                count, value = metric_funcs[metric](pred, tgt)
                train_metrics[metric_key].update(value.item(), count)

        # Compute loss
        # Main loss: flux predictions
        main_count, main_val = predicts.numel(), loss_func(predicts, targets)

        if args.beta > 0:
            # Absorption losses
            abs12_count, abs12_val = (
                abs12_predict.numel(),
                loss_func(abs12_predict, abs12_target),
            )
            abs34_count, abs34_val = (
                abs34_predict.numel(),
                loss_func(abs34_predict, abs34_target),
            )

            weighted_loss = (1.0 - args.beta) * main_val * main_count + args.beta * (
                abs12_val * abs12_count + abs34_val * abs34_count
            )

            total_count = (1.0 - args.beta) * main_count + args.beta * (
                abs12_count + abs34_count
            )

            total_loss = weighted_loss / total_count

        else:
            total_loss = main_val

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()

        # Gradient clipping (optional)
        if hasattr(args, "grad_clip"):
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        optimizer.step()

        # Update trackers
        train_loss_tracker.update(total_loss.item(), 1)

        # Update progress bar
        loop.set_postfix(loss=total_loss.item())

        # TensorBoard logging
        writer.add_scalar(
            "train/total_loss", total_loss.item(), global_step=global_step
        )
        global_step += args.batch_size

    return train_loss_tracker.getmean(), global_step


def train_epoch_cams(
    model,
    train_loader,
    optimizer,
    loss_func,
    metric_funcs,
    metric_names,
    output_keys,
    train_metrics,
    train_loss_tracker,
    norm_mapping,
    normalization_type,
    index_mapping,
    device,
    args,
    epoch,
    writer,
    global_step,
    logger,
):
    """
    Train for one epoch on CAMS atmospheric data.

    Data shapes from dataset (already in B, C, T format):
    - features: (batch, n_sites, n_features, n_layer) -> flattened to (batch*n_sites, n_features, n_layer)
    - targets: (batch, n_sites, n_fluxes, n_level) -> flattened to (batch*n_sites, n_fluxes, n_level)

    Model expects: (batch, features, seq) and returns (batch, output_features, seq)
    """
    model.train()

    # Reset metrics
    for meter in train_metrics.values():
        meter.reset()
    train_loss_tracker.reset()

    loop = tqdm(
        enumerate(train_loader),
        total=len(train_loader),
        desc=f"Epoch {epoch}",
        leave=False,
    )

    for batch_idx, (features, targets, pressure) in loop:
        if epoch == 0 and batch_idx == 0:
            logger.info("First batch shapes (CAMS):")
            logger.info(
                f"  Feature shape: {features.shape}"
            )  # (batch, n_sites, n_features, n_layer)
            logger.info(
                f"  Targets shape: {targets.shape}"
            )  # (batch, n_sites, n_fluxes, n_level)
            logger.info(
                f"  Pressure shape: {pressure.shape}"
            )  # (batch, n_sites, n_layer -1)

        # Reshape inputs: flatten batch and sites dimensions
        feature_shape = features.shape
        target_shape = targets.shape
        pressure_shape = pressure.shape
        inner_batch_size = feature_shape[0] * feature_shape[1]  # batch * n_sites

        # Reshape to (inner_batch, n_features, n_layer)
        features = features.reshape(
            inner_batch_size, feature_shape[2], feature_shape[3]
        ).to(device=device)

        # Reshape to (inner_batch, n_fluxes, n_level)
        targets = targets.reshape(
            inner_batch_size, target_shape[2], target_shape[3]
        ).to(device=device)

        # Reshape pressure to (inner_batch, n_level_interior)
        # pressure has shape (batch, n_sites, 59) -> (inner_batch, 59)
        pressure = pressure.reshape(inner_batch_size, pressure_shape[2]).to(
            device=device
        )

        # Forward pass
        # Model expects: (inner_batch, n_features, n_layer)
        # Model returns: (inner_batch, n_fluxes, n_level)
        predicts = model(features)  # (inner_batch, n_fluxes, n_level)

        # Denormalize predictions and targets
        predicts_unnorm, targets_unnorm = unnorm_mpas(
            predicts,
            targets,
            norm_mapping,
            normalization_type,
            index_mapping,
        )

        assert (
            predicts_unnorm.shape == predicts.shape
        ), f"Expected predicts_unnorm shape {predicts.shape}, got {predicts_unnorm.shape}"
        assert (
            targets_unnorm.shape == targets.shape
        ), f"Expected targets_unnorm shape {targets.shape}, got {targets_unnorm.shape}"

        # Calculate heating rates (HR) from fluxes
        # predicts_unnorm: (inner_batch, n_fluxes, n_level) -> [rsd, rsu]
        # pressure: (inner_batch, n_level_interior) -> (inner_batch, 59)
        hr_predict, hr_target = calc_heating_rates(predicts_unnorm, targets_unnorm)

        if epoch == 0 and batch_idx == 0:
            logger.info(
                f"Heating rate shapes: pred={hr_predict.shape}, target={hr_target.shape}"
            )

        # Compute metrics
        output_dict = {
            "fluxes": (predicts, targets),
            "HR": (hr_predict, hr_target),
        }

        for key in output_keys:
            pred, tgt = output_dict[key]
            for metric in metric_names:
                metric_key = f"{key}_{metric}"
                if metric_key not in train_metrics:
                    logger.error(
                        f"Metric key '{metric_key}' not found in train_metrics"
                    )
                    raise KeyError(
                        f"Metric key '{metric_key}' not found in train_metrics"
                    )
                count, value = metric_funcs[metric](pred, tgt)
                train_metrics[metric_key].update(value.item(), count)

        # Compute loss
        # Main loss: flux predictions
        main_count, main_val = predicts.numel(), loss_func(predicts, targets)

        # Heating rate loss (optional, weighted by beta)
        if args.beta > 0:
            hr_count, hr_val = hr_predict.numel(), loss_func(hr_predict, hr_target)
            weighted_loss = (1.0 - args.beta) * main_val * main_count + args.beta * (
                hr_val * hr_count
            )
            total_count = (1.0 - args.beta) * main_count + args.beta * hr_count
            total_loss = weighted_loss / total_count
        else:
            total_loss = main_val

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()

        # Gradient clipping (optional)
        if hasattr(args, "grad_clip"):
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        optimizer.step()

        # Update trackers
        train_loss_tracker.update(total_loss.item(), 1)

        # Update progress bar
        loop.set_postfix(loss=total_loss.item())

        # TensorBoard logging
        writer.add_scalar(
            "train/total_loss", total_loss.item(), global_step=global_step
        )
        global_step += args.batch_size

    return train_loss_tracker.getmean(), global_step


def main():
    """
    Main entry point for training the RTnn model.
    """
    # Parse arguments
    args = parse_args()

    # Show version and exit if requested
    if args.version:
        print_version()
        return

    # Setup directories and logging
    paths, logger = setup_directories_and_logging(args)

    # Log configuration
    log_configuration(args, paths, logger)

    # Setup metrics
    metric_names = ["NMAE", "NMSE", "R2", "MAE", "MSE"]
    metric_funcs = {
        "NMAE": nmae_all,
        "NMSE": nmse_all,
        "R2": r2_all,
        "MAE": mae_all,
        "MSE": mse_all,
    }

    if args.dataset_type == "ORC":
        n_pft = 15
        n_bands = 2
        n_chans = 4
        output_keys = ["fluxes", "abs12", "abs34"]

        # Index mapping for output variables
        index_mapping = {
            0: "collim_alb",
            1: "collim_tran",
            2: "isotrop_alb",
            3: "isotrop_tran",
        }

        # Get data files
        train_files, test_files = get_data_files_lsm(args, logger)

    elif args.dataset_type == "CAMS_RADSCHEME":
        n_pft = None
        n_bands = None
        n_chans = None
        output_keys = ["fluxes", "HR"]

        # Index mapping for output variables
        index_mapping = {
            0: "rsd",
            1: "rsu",
        }

        # Get data files
        train_files, test_files = get_data_files_rrtmgp(args, logger)

    elif args.dataset_type == "CAMS_REFTRANS":
        n_pft = None
        n_bands = None
        n_chans = None
        output_keys = ["fluxes", "abs12", "abs34"]

        # Index mapping for output variables
        index_mapping = {
            0: "rdif",
            1: "tdif",
            2: "rdir",
            3: "tdir",
        }

        # Get data files
        train_files, test_files = get_data_files_rrtmgp(args, logger)

    else:
        raise ValueError(f"Dataset_type of {args.dataset_type} is not implimented yet!")

    if not train_files:
        raise ValueError("No training files found")
    if not test_files and not args.run_type == "inference":
        logger.warning("No test files found, but continuing with training only")
    if args.run_type == "inference" and not test_files:
        raise ValueError("Inference mode requires test files, but none were found")

    logger.info("Index mapping:")
    for idx, var in index_mapping.items():
        logger.info(f"  {idx} -> '{var}'")

    train_metrics = {
        f"{k}_{m}": MetricTracker() for k in output_keys for m in metric_names
    }
    train_loss_tracker = MetricTracker()

    # Setup device and seed
    device = setup_device_and_seed(args, logger)

    # Compute normalization statistics
    if args.dataset_type == "ORC":
        norm_mapping = create_normalization_mapping_lsm(train_files, paths, logger)
        # Create datasets and loaders
        train_loader, test_loader, normalization_type, train_dataset, test_dataset = (
            create_datasets_and_loaders_lsm(
                args, train_files, test_files, norm_mapping, logger
            )
        )
    elif args.dataset_type == "CAMS_RADSCHEME":
        norm_mapping = create_normalization_mapping_rrtmgp(
            test_files, output_dir=paths.stats, logger=logger
        )
        train_loader, test_loader, normalization_type, train_dataset, test_dataset = (
            create_datasets_and_loaders_rrtmgp(
                args=args,
                train_file=train_files,
                test_file=test_files,
                norm_mapping=norm_mapping,
                logger=logger,
            )
        )
    elif args.dataset_type == "CAMS_REFTRANS":
        norm_mapping = create_normalization_mapping_reftrans(
            test_files, output_dir=paths.stats, logger=logger
        )
        train_loader, test_loader, normalization_type, train_dataset, test_dataset = (
            create_datasets_and_loaders_reftrans(
                args=args,
                train_file=train_files,
                test_file=test_files,
                norm_mapping=norm_mapping,
                logger=logger,
            )
        )
    else:
        raise ValueError(f"Dataset_type of {args.dataset_type} is not implimented yet!")

    # Initialize model
    model = initialize_model(args, device, logger)

    # Setup loss function and optimizer
    loss_func = get_loss_function(args.loss_type, args, logger=logger)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=5
    )

    try:
        # Load checkpoint if requested
        (
            start_epoch,
            samples_processed,
            batches_processed,
            best_val_loss,
            best_epoch,
            checkpoint,
            train_loss_history,
            valid_loss_history,
            valid_metrics_history,
        ) = load_checkpoint_if_requested(args, model, optimizer, paths, device, logger)

        # Setup TensorBoard
        writer = SummaryWriter(paths.runs)
        global_step = samples_processed

        # Run inference only if requested
        if args.run_type == "inference":
            if test_loader is None or len(test_loader) == 0:
                raise ValueError("Inference mode requires test data")

            logger.start_task("Running inference", "Evaluating model on test data")

            if args.dataset_type == "ORC":
                valid_loss, valid_metrics = run_validation_lsm(
                    test_loader,
                    model,
                    norm_mapping,
                    normalization_type,
                    index_mapping,
                    device,
                    args,
                    best_epoch,
                    logger,
                    paths.results,
                    n_pft=n_pft,
                    n_bands=n_bands,
                    n_chans=n_chans,
                )
            elif args.dataset_type == "CAMS_RADSCHEME":
                valid_loss, valid_metrics = run_validation_cams(
                    test_loader,
                    model,
                    norm_mapping,
                    normalization_type,
                    index_mapping,
                    device,
                    args,
                    best_epoch,
                    logger,
                    paths.results,
                )
            elif args.dataset_type == "CAMS_REFTRANS":
                valid_loss, valid_metrics = run_validation_reftrans(
                    test_loader,
                    model,
                    norm_mapping,
                    normalization_type,
                    index_mapping,
                    device,
                    args,
                    best_epoch,
                    logger,
                    paths.results,
                )
            else:
                raise ValueError(
                    f"Dataset_type of {args.dataset_type} is not implimented yet!"
                )

            logger.info("Inference results:")
            logger.info(f"  Validation loss: {valid_loss:.6e}")
            for key, val in valid_metrics.items():
                logger.info(f"  {key}: {val:.6e}")

            logger.success("Inference completed")
            return

        # Initialize or extend history arrays
        if not train_loss_history:
            train_loss_history = [0.0] * args.num_epochs
        else:
            # Extend history if needed
            if len(train_loss_history) < args.num_epochs:
                train_loss_history.extend(
                    [0.0] * (args.num_epochs - len(train_loss_history))
                )

        if not valid_loss_history:
            valid_loss_history = [0.0] * args.num_epochs
        else:
            if len(valid_loss_history) < args.num_epochs:
                valid_loss_history.extend(
                    [0.0] * (args.num_epochs - len(valid_loss_history))
                )

        if not valid_metrics_history:
            valid_metrics_history = {key: [] for key in train_metrics}

        train_metrics_history = {key: [] for key in train_metrics}
        valid_loss = (
            valid_loss_history[start_epoch - 1] if start_epoch > 0 else float("inf")
        )

        # Training loop
        logger.start_task("Starting training", f"Epochs: {args.num_epochs}")

        for epoch in range(start_epoch, args.num_epochs):
            epoch_start_time = time.time()

            # Train for one epoch
            if args.dataset_type == "ORC":
                avg_train_loss, global_step = train_epoch_lsm(
                    model,
                    train_loader,
                    optimizer,
                    loss_func,
                    metric_funcs,
                    metric_names,
                    output_keys,
                    train_metrics,
                    train_loss_tracker,
                    norm_mapping,
                    normalization_type,
                    index_mapping,
                    device,
                    args,
                    epoch,
                    writer,
                    global_step,
                    logger,
                    n_pft=n_pft,
                    n_bands=n_bands,
                    n_chans=n_chans,
                )
            elif args.dataset_type == "CAMS_RADSCHEME":
                avg_train_loss, global_step = train_epoch_cams(
                    model,
                    train_loader,
                    optimizer,
                    loss_func,
                    metric_funcs,
                    metric_names,
                    output_keys,
                    train_metrics,
                    train_loss_tracker,
                    norm_mapping,
                    normalization_type,
                    index_mapping,
                    device,
                    args,
                    epoch,
                    writer,
                    global_step,
                    logger,
                )

            elif args.dataset_type == "CAMS_REFTRANS":
                avg_train_loss, global_step = train_epoch_reftrans(
                    model,
                    train_loader,
                    optimizer,
                    loss_func,
                    metric_funcs,
                    metric_names,
                    output_keys,
                    train_metrics,
                    train_loss_tracker,
                    norm_mapping,
                    normalization_type,
                    index_mapping,
                    device,
                    args,
                    epoch,
                    writer,
                    global_step,
                    logger,
                )
            else:
                raise ValueError(
                    f"Dataset_type of {args.dataset_type} is not implimented yet!"
                )
            train_loss_history[epoch] = avg_train_loss

            # Validation
            if test_loader and len(test_loader) > 0:
                if args.dataset_type == "ORC":
                    valid_loss, valid_metrics = run_validation_lsm(
                        test_loader,
                        model,
                        norm_mapping,
                        normalization_type,
                        index_mapping,
                        device,
                        args,
                        epoch,
                        logger,
                        paths.results,
                        n_pft=n_pft,
                        n_bands=n_bands,
                        n_chans=n_chans,
                    )
                elif args.dataset_type == "CAMS_RADSCHEME":
                    valid_loss, valid_metrics = run_validation_cams(
                        test_loader,
                        model,
                        norm_mapping,
                        normalization_type,
                        index_mapping,
                        device,
                        args,
                        epoch,
                        logger,
                        paths.results,
                    )
                elif args.dataset_type == "CAMS_REFTRANS":
                    valid_loss, valid_metrics = run_validation_reftrans(
                        test_loader,
                        model,
                        norm_mapping,
                        normalization_type,
                        index_mapping,
                        device,
                        args,
                        epoch,
                        logger,
                        paths.results,
                    )
                else:
                    raise ValueError(
                        f"Dataset_type of {args.dataset_type} is not implimented yet!"
                    )

                valid_loss_history[epoch] = valid_loss

                # Log metrics
                logger.info(
                    f"Epoch {epoch:3d} | "
                    f"Train Loss: {avg_train_loss:.6e} | "
                    f"Valid Loss: {valid_loss:.6e} | "
                    f"Time: {time.time() - epoch_start_time:.1f}s"
                )

                # Update metrics history
                for key in train_metrics:
                    train_value = train_metrics[key].getmean()
                    valid_value = valid_metrics.get(key, 0.0)

                    train_metrics_history[key].append(train_value)
                    if key in valid_metrics_history:
                        valid_metrics_history[key].append(valid_value)
                    else:
                        valid_metrics_history[key] = [valid_value]

                    logger.info(
                        f"  {key}: train={train_value:.6e}, valid={valid_value:.6e}"
                    )

                # Update scheduler
                scheduler.step(valid_loss)

                # Save best model
                if valid_loss < best_val_loss:
                    best_val_loss = valid_loss
                    best_epoch = epoch
                    if args.save_model:
                        ModelUtils.save_training_checkpoint(
                            model=model,
                            optimizer=optimizer,
                            epoch=epoch,
                            samples_processed=samples_processed,
                            batches_processed=batches_processed,
                            train_loss_history=train_loss_history,
                            valid_loss_history=valid_loss_history,
                            valid_metrics_history=valid_metrics_history,
                            best_val_loss=best_val_loss,
                            best_epoch=best_epoch,
                            avg_val_loss=valid_loss,
                            avg_epoch_loss=avg_train_loss,
                            args=args,
                            paths=paths,
                            logger=logger,
                            checkpoint_type="best",
                            save_full_model=True,
                        )

            # Save epoch checkpoint (every 10 epochs)
            if args.save_model and epoch % 10 == 0:
                ModelUtils.save_training_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    samples_processed=samples_processed,
                    batches_processed=batches_processed,
                    train_loss_history=train_loss_history,
                    valid_loss_history=valid_loss_history,
                    valid_metrics_history=valid_metrics_history,
                    best_val_loss=best_val_loss,
                    best_epoch=best_epoch,
                    avg_val_loss=valid_loss if test_loader else 0.0,
                    avg_epoch_loss=avg_train_loss,
                    args=args,
                    paths=paths,
                    logger=logger,
                    checkpoint_type="epoch",
                    save_full_model=True,
                )

        # Generate plots at the end of training
        logger.info("Generating training summary plots...")

        plot_metric_histories(
            train_metrics_history,
            valid_metrics_history,
            os.path.join(paths.results, "metrics_panel.png"),
        )
        plot_loss_histories(
            train_loss_history[: args.num_epochs],
            valid_loss_history[: args.num_epochs],
            os.path.join(paths.results, "training_validation_loss.png"),
        )

        plot_spatial_temporal_density(
            sindex_tracker=train_dataset.sindex_tracker,
            tindex_tracker=train_dataset.tindex_tracker,
            mode="train",
            save_dir=paths.results,
            filename="density_scatter_train",
        )

        plot_spatial_temporal_density(
            sindex_tracker=test_dataset.sindex_tracker,
            tindex_tracker=test_dataset.tindex_tracker,
            mode="test",
            save_dir=paths.results,
            filename="density_scatter_test",
        )

        # Save final checkpoint
        if args.save_model:
            ModelUtils.save_training_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=args.num_epochs - 1,
                samples_processed=samples_processed,
                batches_processed=batches_processed,
                train_loss_history=train_loss_history,
                valid_loss_history=valid_loss_history,
                valid_metrics_history=valid_metrics_history,
                best_val_loss=best_val_loss,
                best_epoch=best_epoch,
                avg_val_loss=valid_loss if test_loader else 0.0,
                avg_epoch_loss=avg_train_loss if "avg_train_loss" in locals() else 0.0,
                args=args,
                paths=paths,
                logger=logger,
                checkpoint_type="final",
                save_full_model=True,
            )
            logger.info("Final model checkpoint saved successfully!")

        logger.success("Training completed successfully!")

    except Exception as e:
        logger.exception("Training failed with exception", e)
        raise


if __name__ == "__main__":
    main()
