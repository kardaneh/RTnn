import argparse
import os
import logging
import datetime
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import xarray as xr

from file_helper import FileUtils
from data_helper import DataPreprocessor
from model_prepare import load_model
from model_helper import ModelUtils
from evaluate_helper import (
    unnorm_mpas,
    get_hr,
    check_accuracy_evaluate,
    MetricTracker,
)
from plot_helper import plot_metric_histories

def parse_args():
    parser = argparse.ArgumentParser(description="Train the RTM model")

    # Files and Directories
    parser.add_argument(
        "--nc_file", type=str, default="rrtmg4nn.nc", help="NetCDF file"
    )
    parser.add_argument("--root_dir", type=str, default="", help="Root directory")
    parser.add_argument(
        "--train_file", type=str, default="", help="Path to training dataset"
    )
    parser.add_argument(
        "--test_file", type=str, default="", help="Path to test dataset"
    )
    parser.add_argument(
        "--train_point_number", type=int, default=100, help="Number of training points"
    )
    parser.add_argument(
        "--test_point_number", type=int, default=100, help="Number of testing points"
    )

    # Directory structure
    parser.add_argument(
        "--main_folder", type=str, default="temp", help="Main folder name"
    )
    parser.add_argument(
        "--sub_folder", type=str, default="temp", help="Sub-folder name"
    )
    parser.add_argument("--prefix", type=str, default="temp", help="Prefix for saving")

    # Model parameters
    parser.add_argument(
        "--dataset_type", type=str, default="Large", help="Type of dataset"
    )
    parser.add_argument(
        "--loss_type", type=str, default="v01", help="Loss function type"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-3, help="Learning rate"
    )

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=200, help="Batch size")
    parser.add_argument("--model_name", type=str, default="FC", help="Model name")
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of workers for data loading"
    )
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")

    # Checkpointing options
    parser.add_argument(
        "--save_model",
        choices=("True", "False"),
        default="False",
        help="Save the trained model",
    )
    parser.add_argument(
        "--save_checkpoint_name",
        type=str,
        default="test.pth.tar",
        help="Checkpoint file name",
    )
    parser.add_argument(
        "--save_per_samples",
        type=int,
        default=10000,
        help="Frequency of saving checkpoints",
    )

    parser.add_argument(
        "--load_model",
        choices=("True", "False"),
        default="False",
        help="Load a pre-trained model",
    )
    parser.add_argument(
        "--load_checkpoint_name",
        type=str,
        default="test.pth.tar",
        help="Checkpoint file to load",
    )

    # Additional options
    parser.add_argument(
        "--random_throw",
        choices=("True", "False"),
        default="False",
        help="Random throw option",
    )
    parser.add_argument(
        "--only_layer",
        choices=("True", "False"),
        default="False",
        help="Use only a specific layer",
    )

    args = parser.parse_args()
    return args


args = parse_args()
full_file_path = os.path.join(args.root_dir, args.train_file)
df = xr.open_dataset(full_file_path, engine="netcdf4")

from torch.utils.tensorboard import SummaryWriter

FileUtils.makedir(os.path.join("logs", args.main_folder, args.sub_folder))
FileUtils.makedir(os.path.join("results", args.main_folder, args.sub_folder))
FileUtils.makedir(os.path.join("runs", args.main_folder, args.sub_folder))
FileUtils.makedir(os.path.join("checkpoints", args.main_folder, args.sub_folder))

if args.random_throw == "True":
    args.random_throw_boolean = True
else:
    args.random_throw_boolean = False

if args.only_layer == "True":
    args.only_layer_boolean = True
else:
    args.only_layer_boolean = False

# Create a FileHandler to log the output to a file
now = datetime.datetime.now()
date_time_str = now.strftime("%Y%m%d_%H%M%S")
log_dir = os.path.join("logs", args.main_folder, args.sub_folder)
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"{args.prefix}_{date_time_str}_log.txt")

logger = logging.getLogger("")
logger.setLevel(logging.INFO)

if logger.hasHandlers():
    logger.handlers.clear()

file_handler = logging.FileHandler(log_file, mode='w')
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
step = (0)
logger.info(f"NetCDF file: {full_file_path}")
norm_mapping = {
        var: {
            'mean': df[var].mean().item(),
            'std': df[var].std().item()
            }
        for var in df.data_vars
        }

# Create training dataset
train_dataset = DataPreprocessor(
        logger = logger,
        df = df,
        from_time=0,
        end_time=1,
        batch_divid_number=160,
        point_folds=1,
        time_folds=1,
        norm_mapping=norm_mapping,
        point_number=args.train_point_number,
        only_layer=args.only_layer_boolean
        )

# Create testing dataset
test_dataset = DataPreprocessor(
        logger = logger,
        df=df,
        from_time=0,
        end_time=1,
        batch_divid_number=160,
        point_folds=1,
        time_folds=1,
        norm_mapping=norm_mapping,
        point_number=args.test_point_number,
        only_layer=args.only_layer_boolean
        )

# Create DataLoader for training
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
    pin_memory=False,
)

# Create DataLoader for testing
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
    pin_memory=False,
)

# ---------------------------------------------
# Dataset Information
# ---------------------------------------------
logger.info(f"Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")

# ---------------------------------------------
# Model Initialization
# ---------------------------------------------
model = load_model(model_name=args.model_name, device=device, feature_channel=34, signal_length=57)

model_info = ModelUtils.get_parameter_number(model)
logger.info(f"Model Info: {model_info}")

model = model.to(device)

# ---------------------------------------------
# Loss Functions & Optimizer Setup
# ---------------------------------------------
criterion_mse = nn.MSELoss()
criterion_mae = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5, verbose=True)

# ---------------------------------------------
# Load Checkpoint (if specified)
# ---------------------------------------------
if args.load_model == "True":
    checkpoint_path = os.path.join("checkpoints", args.main_folder, args.sub_folder, args.load_checkpoint_name)
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
# Output Index Mapping & Training Start
# ---------------------------------------------
index_mapping = {0: "swuflx", 1: "swdflx", 2: "lwuflx", 3: "lwdflx"}
logger.info("Start training...")

#norm_mapping = train_dataset.stats

# Training metrics
train_metrics = {
    'rmse': MetricTracker(),
    'mae': MetricTracker(),
    'swhr_rmse': MetricTracker(),
    'lwhr_rmse': MetricTracker(),
    'swhr_mae': MetricTracker(),
    'lwhr_mae': MetricTracker(),
}

train_metrics_history = {key: [] for key in train_metrics}
valid_metrics_history = {key: [] for key in train_metrics}

for epoch in range(args.num_epochs):
    model.train()

    for meter in train_metrics.values():
        meter.reset()

    schedule_losses = []
    previous_time = time.time()

    loop = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}")
    for batch_idx, (feature, targets, auxis) in loop:
        if epoch == 0 and batch_idx == 0:
            logger.info(f"feature shape:{feature.shape}, target shape:{targets.shape}, auxis shape:{auxis.shape}")

        feature_shape = feature.shape
        target_shape = targets.shape
        auxis_shape = auxis.shape

        inner_batch_size = feature_shape[0] * feature_shape[1]
        feature = feature.reshape(inner_batch_size, feature_shape[2], feature_shape[3]).to(device=device)
        targets = targets.reshape(inner_batch_size, target_shape[2], target_shape[3]).to(device=device)
        auxis = auxis.reshape(inner_batch_size, auxis_shape[2], auxis_shape[3]).to(device=device)
        
        predicts = model(feature)

        predicts_unnorm, targets_unnorm = unnorm_mpas(predicts, targets, norm_mapping, index_mapping)
        swhr_predict, swhr_target, lwhr_predict, lwhr_target = get_hr(predicts_unnorm, targets_unnorm, auxis)
        
        metric_values = {
                'rmse': criterion_mse(predicts, targets),
                'mae': criterion_mae(predicts, targets),
                'swhr_rmse': criterion_mse(swhr_predict, swhr_target),
                'lwhr_rmse': criterion_mse(lwhr_predict, lwhr_target),
                'swhr_mae': criterion_mae(swhr_predict, swhr_target),
                'lwhr_mae': criterion_mae(lwhr_predict, lwhr_target)
                }

        beta = 0.001
        total_loss = metric_values['rmse'] #+ beta * (metric_values['swhr_rmse'] + metric_values['lwhr_rmse'])
        loop.set_postfix(loss=total_loss.item())

        for key, value in metric_values.items():
            train_metrics[key].update(value.item(), feature_shape[0])

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        writer.add_scalar("train_mse", metric_values['rmse'].item(), global_step=step)
        writer.add_scalar("train_mae", metric_values['mae'].item(), global_step=step)

        step = step + args.batch_size

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
    logger.info(f"elapse time:{time.time() - previous_time}")
    if epoch % 1 == 0:

        valid_metrics = check_accuracy_evaluate(
                        test_loader,
                        model,
                        norm_mapping,
                        index_mapping,
                        device,
                        args,
                        if_plot=False,
                        target_norm_info = None
                        )

        for key, meter in train_metrics.items():
            train_value = meter.getsqrtmean() if 'rmse' in key else meter.getmean()
            train_metrics_history[key].append(train_value)

            assert key in valid_metrics, f"Missing key '{key}' in valid_metrics"
            
            valid_value = valid_metrics[key]
            valid_metrics_history[key].append(valid_value)

            logger.info(f"train_{key}: {train_value:.3e} | valid_{key}: {valid_value:.3e}")

        logger.info("")
        schedule_losses.append(valid_metrics['swhr_rmse'] + valid_metrics['lwhr_rmse'])
        mean_loss = sum(schedule_losses) / len(schedule_losses)
        scheduler.step(mean_loss)

base_dir = os.path.join("results", args.main_folder, args.sub_folder)
plot_metric_histories(train_metrics_history, valid_metrics_history, os.path.join(base_dir, f"metrics_panel.png"))
