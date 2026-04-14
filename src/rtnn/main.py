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
"""

import os
import time
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Import local modules
from rtnn.utils import FileUtils, EasyDict
from rtnn.dataset import DataPreprocessor
from rtnn.model_loader import load_model
from rtnn.model_utils import ModelUtils
from rtnn.evaluater import (
    unnorm_mpas,
    calc_abs,
    run_validation,
    get_loss_function,
    MetricTracker,
    r2_all,
    nmae_all,
    nmse_all,
)
from rtnn.diagnostics import (
    plot_metric_histories,
    plot_loss_histories,
    plot_spatial_temporal_density,
    stats,
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
                Dataset type (e.g., "LSM", "RTM").

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
        default="LSM",
        choices=["LSM", "RTM"],
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
        choices=["lstm", "gru", "fcn", "fullyconnected", "transformer", "cnn", "mlp"],
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


def get_data_files(args, logger):
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


def create_normalization_mapping(train_files, paths, logger):
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


def create_datasets_and_loaders(args, train_files, test_files, norm_mapping, logger):
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
    test_tbatch = 1 if args.run_type == "inference" else 48
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
    start_epoch = epoch + 1 if args.resume_training else epoch

    if args.resume_training:
        logger.info(f"Resuming training from epoch {start_epoch}")
        if args.debug:
            logger.info(f"Current train_loss_history length: {len(train_loss_history)}")
            logger.info(f"Current valid_loss_history length: {len(valid_loss_history)}")
    else:
        logger.info(
            f"Model loaded for 'inference' if {args.run_type} == 'inference' else 'continued training'"
        )

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


def save_checkpoint(
    model,
    optimizer,
    epoch,
    samples_processed,
    batches_processed,
    train_loss,
    valid_loss,
    args,
    paths,
    logger,
    checkpoint_type="epoch",
):
    """
    Save model checkpoint using ModelUtils.

    Parameters
    ----------
    model : torch.nn.Module
        Model to save.
    optimizer : torch.optim.Optimizer
        Optimizer state to save.
    epoch : int
        Current epoch.
    samples_processed : int
        Total samples processed.
    batches_processed : int
        Total batches processed.
    train_loss : float
        Current training loss.
    valid_loss : float
        Current validation loss.
    args : argparse.Namespace
        Command-line arguments.
    paths : EasyDict
        Directory paths.
    logger : Logger
        Logger instance.
    checkpoint_type : str
        Type of checkpoint ("epoch", "best", "final", "samples").
    """
    # Prepare history data (simplified for this example)
    train_loss_history = [0.0] * (epoch + 1)
    train_loss_history[epoch] = train_loss
    valid_loss_history = [0.0] * (epoch + 1)
    valid_loss_history[epoch] = valid_loss

    ModelUtils.save_training_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=epoch,
        samples_processed=samples_processed,
        batches_processed=batches_processed,
        train_loss_history=train_loss_history,
        valid_loss_history=valid_loss_history,
        valid_metrics_history={},  # Will be populated in full implementation
        best_val_loss=valid_loss,
        best_epoch=epoch,
        avg_val_loss=valid_loss,
        avg_epoch_loss=train_loss,
        args=args,
        paths=paths,
        logger=logger,
        checkpoint_type=checkpoint_type,
        save_full_model=True,
    )


def train_epoch(
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

        # Unnormalize for absorption calculation
        predicts_unnorm, targets_unnorm = unnorm_mpas(
            predicts, targets, norm_mapping, normalization_type, index_mapping
        )

        # Calculate absorption
        abs12_predict, abs12_target, abs34_predict, abs34_target = calc_abs(
            predicts_unnorm, targets_unnorm
        )

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

    try:
        # Setup device and seed
        device = setup_device_and_seed(args, logger)

        # Get data files
        train_files, test_files = get_data_files(args, logger)

        if not train_files:
            raise ValueError("No training files found")
        if not test_files and not args.run_type == "inference":
            logger.warning("No test files found, but continuing with training only")
        if args.run_type == "inference" and not test_files:
            raise ValueError("Inference mode requires test files, but none were found")

        # Compute normalization statistics
        norm_mapping = create_normalization_mapping(train_files, paths, logger)

        # Create datasets and loaders
        train_loader, test_loader, normalization_type, train_dataset, test_dataset = (
            create_datasets_and_loaders(
                args, train_files, test_files, norm_mapping, logger
            )
        )

        # Initialize model
        model = initialize_model(args, device, logger)

        # Setup loss function and optimizer
        loss_func = get_loss_function(args.loss_type, args, logger=logger)
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=5
        )

        # Setup metrics
        metric_names = ["NMAE", "NMSE", "R2"]
        metric_funcs = {"NMAE": nmae_all, "NMSE": nmse_all, "R2": r2_all}
        output_keys = ["fluxes", "abs12", "abs34"]

        train_metrics = {
            f"{k}_{m}": MetricTracker() for k in output_keys for m in metric_names
        }
        train_loss_tracker = MetricTracker()

        # Index mapping for output variables
        index_mapping = {
            0: "collim_alb",
            1: "collim_tran",
            2: "isotrop_alb",
            3: "isotrop_tran",
        }
        logger.info("Index mapping initialized.")
        for idx, var in index_mapping.items():
            logger.info(f"  {idx} -> '{var}'")
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

            valid_loss, valid_metrics = run_validation(
                test_loader,
                model,
                norm_mapping,
                normalization_type,
                index_mapping,
                device,
                args,
                args.num_epochs - 1,
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

        # Training loop
        logger.start_task("Starting training", f"Epochs: {args.num_epochs}")

        for epoch in range(start_epoch, args.num_epochs):
            epoch_start_time = time.time()

            # Train for one epoch
            avg_train_loss, global_step = train_epoch(
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

            train_loss_history[epoch] = avg_train_loss

            # Validation
            if test_loader and len(test_loader) > 0:
                valid_loss, valid_metrics = run_validation(
                    test_loader,
                    model,
                    norm_mapping,
                    normalization_type,
                    index_mapping,
                    device,
                    args,
                    epoch,
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
            filename="density_scatter_test",
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
