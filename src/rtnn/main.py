import argparse
import os
import glob
import logging
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import xarray as xr

from rtnn.utils import FileUtils
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
from rtnn.diagnostics import plot_metric_histories, plot_loss_histories, stats
from rtnn import __version__, __version_info__, __author__, __license__, __copyright__
from torch.utils.tensorboard import SummaryWriter


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
        choices=["mse", "mae", "nmae", "nmse", "wmse", "logcosh", "smoothl1", "huber"],
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
        "--tbatch", type=int, default=24, help="Temporal batch length for processing"
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
        "--load_model",
        type=lambda x: x.lower() == "true",
        default=False,
        help="Load existing model",
    )

    parser.add_argument(
        "--inference",
        type=lambda x: x.lower() == "true",
        default=False,
        help="Run in inference-only mode",
    )

    parser.add_argument(
        "--load_checkpoint_name",
        type=str,
        default="model.pth.tar",
        help="Checkpoint file to load",
    )

    args = parser.parse_args()
    return args


def main():
    """
    Main entry point for training the RTnn model.

    This function orchestrates the complete training pipeline for the Radiative
    Transfer Neural Network model. It handles:

    - Parsing command-line arguments for model configuration, training parameters,
      and data paths
    - Loading and preprocessing NetCDF climate data
    - Initializing the specified model architecture (LSTM, GRU, Transformer, FCN, etc.)
    - Setting up optimizers, loss functions, and learning rate schedulers
    - Running the training loop with progress tracking and validation
    - Saving model checkpoints and logging metrics
    - Generating visualization plots for training history

    Parameters
    ----------
    args : argparse.Namespace, optional
        Command-line arguments. If None, arguments are parsed from sys.argv.

    Returns
    -------
    None

    Notes
    -----
    The training process includes:
    - Random seed initialization for reproducibility
    - GPU support with automatic device detection
    - TensorBoard logging for monitoring training progress
    - Combined loss function (main loss + absorption loss)
    - Learning rate scheduling based on validation performance
    - Periodic model checkpointing
    - Final plotting of training and validation metrics

    Examples
    --------
    >>> # Run training with default parameters
    >>> main()

    >>> # With custom arguments (typically via command line)
    >>> main(parse_args())

    Command line usage:
    $ rtnn --type lstm --hidden_size 128 --num_layers 3 --batch_size 32
    """
    args = parse_args()

    if args.version:
        print_version()
        return

    train_years = parse_years(args.train_years)
    # train_sbatch_files = np.sort(glob.glob(args.train_data_files + f"rtnetcdf_*_{args.train_year}.nc"))[::]
    train_sbatch_files = sorted(
        file
        for year in train_years
        for file in glob.glob(f"{args.train_data_files}/rtnetcdf_*_{year}.nc")
    )
    test_sbatch_files = np.sort(
        glob.glob(args.test_data_files + f"rtnetcdf_*_{args.test_year}.nc")
    )[::]
    train_df = xr.open_dataset(train_sbatch_files[0], engine="netcdf4")

    FileUtils.makedir(os.path.join("logs", args.main_folder, args.sub_folder))
    FileUtils.makedir(os.path.join("results", args.main_folder, args.sub_folder))
    FileUtils.makedir(os.path.join("runs", args.main_folder, args.sub_folder))
    FileUtils.makedir(os.path.join("checkpoints", args.main_folder, args.sub_folder))
    FileUtils.makedir(os.path.join("stats", args.main_folder, args.sub_folder))

    # Create a FileHandler to log the output to a file
    # now = datetime.datetime.now()
    # date_time_str = now.strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join("logs", args.main_folder, args.sub_folder)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{args.prefix}_log.txt")

    logger = logging.getLogger("")
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(logging.INFO)

    # logs to console
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)

    # Add both handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    # Disable matplotlib font manager warnings
    logging.getLogger("matplotlib.font_manager").disabled = True

    # Set the random seed for NumPy to ensure reproducibility
    random_state = 0
    np.random.seed(random_state)
    torch.manual_seed(random_state)

    torch.set_printoptions(precision=5)

    # Set device for model (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Enable anomaly detection for debugging gradients
    torch.autograd.set_detect_anomaly(True)

    # Set up TensorBoard logging (useful for visualizing training progress)
    writer = SummaryWriter(f"runs/{args.main_folder}/{args.sub_folder}/")

    # Initialize the step counter for TensorBoard logging or training steps
    index_mapping = {
        0: "collim_alb",
        1: "collim_tran",
        2: "isotrop_alb",
        3: "isotrop_tran",
    }
    step = 0
    logger.info(f"NetCDF files path: {args.train_data_files}")
    logger.info(f"NetCDF files path: {args.test_data_files}")
    logger.info(f"Found {len(train_sbatch_files)} training files:")
    for f in train_sbatch_files:
        logger.info(f"  {f}")
    logger.info(f"Found {len(test_sbatch_files)} test files:")
    for f in test_sbatch_files:
        logger.info(f"  {f}")

    norm_mapping = stats(
        [train_sbatch_files[0]],
        logger,
        os.path.join("stats", args.main_folder, args.sub_folder),
    )
    for var_name, stats_dict in norm_mapping.items():
        logger.info(f"Variable: {var_name}")
        for stat_key, value in stats_dict.items():
            logger.info(f"  {stat_key}: {value:.4e}")

    normalization_type = {
        "coszang": "log1p_standard",
        "laieff_collim": "log1p_standard",
        "laieff_isotrop": "log1p_standard",
        "leaf_ssa": "log1p_standard",
        "leaf_psd": "log1p_standard",
        "rs_surface_emu": "log1p_standard",
        "collim_alb": "log1p_standard",
        "collim_tran": "log1p_standard",
        "isotrop_alb": "log1p_standard",
        "isotrop_tran": "log1p_standard",
    }

    # Create training dataset
    train_dataset = DataPreprocessor(
        logger=logger,
        dfs=train_sbatch_files,
        stime=0,
        tstep=train_df.sizes["time"],
        tbatch=args.tbatch,
        norm_mapping=norm_mapping,
        normalization_type=normalization_type,
    )

    test_tbatch = 1 if args.inference == "True" else 24
    test_dataset = DataPreprocessor(
        logger=logger,
        dfs=test_sbatch_files,
        stime=0,
        tstep=train_df.sizes["time"],
        tbatch=test_tbatch,
        norm_mapping=norm_mapping,
        normalization_type=normalization_type,
    )

    # Create DataLoader for training
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Create DataLoader for testing
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # ---------------------------------------------
    # Dataset Information
    # ---------------------------------------------
    logger.info(f"Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")

    # ---------------------------------------------
    # Model Initialization
    # ---------------------------------------------
    model = load_model(args)
    model_info = ModelUtils.get_parameter_number(model)
    logger.info(f"Model Info: {model_info}")
    model = model.to(device)

    # ---------------------------------------------
    # Loss Functions & Optimizer Setup
    # ---------------------------------------------
    valid_loss_types = [
        "mse",
        "mae",
        "nmae",
        "nmse",
        "wmse",
        "logcosh",
        "smoothl1",
        "huber",
    ]
    loss_type = args.loss_type.lower()
    assert (
        loss_type in valid_loss_types
    ), f"Invalid loss_type (should be one of {valid_loss_types})"

    func = get_loss_function(loss_type, args)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=5
    )

    metric_names = ["NMAE", "NMSE", "R2"]
    metric_funcs = {"NMAE": nmae_all, "NMSE": nmse_all, "R2": r2_all}
    output_keys = ["fluxes", "abs12", "abs34"]
    train_metrics = {
        f"{k}_{m}": MetricTracker() for k in output_keys for m in metric_names
    }

    # ---------------------------------------------
    # Load Checkpoint (if specified)
    # ---------------------------------------------
    if args.load_model == "True":
        checkpoint_path = os.path.join(
            "checkpoints", args.main_folder, args.sub_folder, args.load_checkpoint_name
        )
        checkpoint = torch.load(checkpoint_path)
        ModelUtils.load_checkpoint(checkpoint, model, optimizer)

    # ---------------------------------------------
    # Prepare for Saving Model
    # ---------------------------------------------
    if args.save_model == "True":
        save_counter = 0

    # ---------------------------------------------
    # GPU Support
    # ---------------------------------------------
    if torch.cuda.is_available():
        model.cuda()
        if torch.cuda.device_count() > 1:
            logger.info(f"Using {torch.cuda.device_count()} GPUs!")
            model = nn.DataParallel(model)

    # ---------------------------------------------
    # Evaluation Only Mode
    # ---------------------------------------------
    if args.inference == "True":
        if not args.load_model == "True":
            raise ValueError(
                "In inference mode, --load_model must be set to 'True' to load the model for evaluation."
            )
        logger.info("Inference mode enabled. Skipping training...")
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
        logger.info(f"Inference valid_loss: {valid_loss:.3e}")
        for key, val in valid_metrics.items():
            logger.info(f"Inference {key}: {val:.3e}")
        exit(0)

    # ---------------------------------------------
    # Output Index Mapping & Training Start
    # ---------------------------------------------
    logger.info("Start training...")

    train_metrics_history = {key: [] for key in train_metrics}
    valid_metrics_history = {key: [] for key in train_metrics}
    train_loss_history = [0] * args.num_epochs
    valid_loss_history = [0] * args.num_epochs
    train_loss = MetricTracker()

    for epoch in range(args.num_epochs):
        model.train()
        for meter in train_metrics.values():
            meter.reset()
        train_loss.reset()
        schedule_losses = []
        previous_time = time.time()

        loop = tqdm(
            enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}"
        )
        for batch_idx, (feature, targets) in loop:
            if epoch == 0 and batch_idx == 0:
                logger.info(
                    f"batch idx:{batch_idx}, feature shape:{feature.shape}, target shape:{targets.shape}"
                )

            feature_shape = feature.shape
            target_shape = targets.shape

            inner_batch_size = feature_shape[0] * feature_shape[1]
            feature = feature.reshape(
                inner_batch_size, feature_shape[2], feature_shape[3]
            ).to(device=device)
            targets = targets.reshape(
                inner_batch_size, target_shape[2], target_shape[3]
            ).to(device=device)

            predicts = model(feature)

            predicts_unnorm, targets_unnorm = unnorm_mpas(
                predicts, targets, norm_mapping, normalization_type, index_mapping
            )
            abs12_predict, abs12_target, abs34_predict, abs34_target = calc_abs(
                predicts_unnorm, targets_unnorm
            )

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
                        raise KeyError(
                            f"Metric key '{metric_key}' not found in train_metrics"
                        )
                    count, value = metric_funcs[metric](pred, tgt)
                    train_metrics[metric_key].update(value.item(), count)

            main_count, main_val = predicts.numel(), func(predicts, targets)
            abs12_count, abs12_val = (
                abs12_predict.numel(),
                func(abs12_predict, abs12_target),
            )
            abs34_count, abs34_val = (
                abs34_predict.numel(),
                func(abs34_predict, abs34_target),
            )

            weighted_loss = (1.0 - args.beta) * main_val * main_count + args.beta * (
                abs12_val * abs12_count + abs34_val * abs34_count
            )
            total_count = (1.0 - args.beta) * main_count + args.beta * (
                abs12_count + abs34_count
            )
            total_loss = weighted_loss / total_count
            loop.set_postfix(loss=total_loss.item())
            train_loss.update(total_loss.item(), 1)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            writer.add_scalar("train/total_loss", total_loss.item(), global_step=step)
            step += args.batch_size

            if args.save_model == "True":
                save_counter = save_counter + args.batch_size
                if save_counter > args.save_per_samples:
                    if torch.cuda.device_count() > 1:
                        checkpoint = {
                            "state_dict": model.module.state_dict(),
                            "optimizer": optimizer.state_dict(),
                        }
                    else:
                        checkpoint = {
                            "state_dict": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                        }

                    filename = os.path.join(
                        "checkpoints",
                        args.main_folder,
                        args.sub_folder,
                        args.prefix
                        + str(epoch).zfill(4)
                        + args.save_checkpoint_name
                        + ".pth.tar",
                    )

                    filename_full = os.path.join(
                        "checkpoints",
                        args.main_folder,
                        args.sub_folder,
                        args.prefix
                        + str(epoch).zfill(4)
                        + args.save_checkpoint_name
                        + ".pth",
                    )

                    ModelUtils.save_checkpoint(checkpoint, filename=filename)
                    if torch.cuda.device_count() > 1:
                        torch.save(model.module, filename_full)
                    else:
                        torch.save(model, filename_full)

                    save_counter = 0
        logger.info(f"Epoch {epoch} elapsed time: {time.time() - previous_time:.2f}s")
        train_loss_history[epoch] = train_loss.getmean()
        if epoch % 1 == 0:
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
            logger.info(
                f"train_loss: {train_loss_history[epoch]:.3e} | valid_loss: {valid_loss_history[epoch]:.3e}"
            )
            for key, meter in train_metrics.items():
                train_value = (
                    meter.getsqrtmean()
                    if key.lower().endswith("mse")
                    else meter.getmean()
                )
                train_metrics_history[key].append(train_value)

                assert key in valid_metrics, f"Missing key '{key}' in valid_metrics"

                valid_value = valid_metrics[key]
                valid_metrics_history[key].append(valid_value)

                logger.info(
                    f"train_{key}: {train_value:.3e} | valid_{key}: {valid_value:.3e}"
                )

            logger.info("")
            schedule_losses.append(valid_metrics["fluxes_NMAE"])
            mean_loss = sum(schedule_losses) / len(schedule_losses)
            scheduler.step(mean_loss)

    base_dir = os.path.join("results", args.main_folder, args.sub_folder)
    plot_metric_histories(
        train_metrics_history,
        valid_metrics_history,
        os.path.join(base_dir, "metrics_panel.png"),
    )
    plot_loss_histories(
        train_loss_history,
        valid_loss_history,
        os.path.join(base_dir, "training_validation_loss.png"),
    )


if __name__ == "__main__":
    main()
