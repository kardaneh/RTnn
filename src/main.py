import logging
import os
import argparse

os.environ["HDF5_DISABLE_VERSION_CHECK"] = "2"
from file_helper import FileUtils
from data_helper import RtmMpasDatasetWholeTimeLarge
from model_prepare import load_model
from model_helper import ModelUtils
from evaluate_helper import (
    unnormalized_mpas,
    MSELoss_all,
    get_heat_rate,
    check_accuracy,
)
from config import norm_mapping, norm_mapping_fullyear_new
import numpy as np
import time

from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch.optim as optim


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
FileUtils.makedir(os.path.join("../logs", args.main_folder, args.sub_folder))
FileUtils.makedir(os.path.join("../results", args.main_folder, args.sub_folder))
FileUtils.makedir(os.path.join("../runs", args.main_folder, args.sub_folder))
FileUtils.makedir(os.path.join("../checkpoints", args.main_folder, args.sub_folder))

if args.random_throw == "True":
    args.random_throw_boolean = True
else:
    args.random_throw_boolean = False

if args.only_layer == "True":
    args.only_layer_boolean = True
else:
    args.only_layer_boolean = False

# Create a FileHandler to log the output to a file
# This will create a log file inside the 'logs' directory, under subdirectories based on the 'main_folder', 'sub_folder' and a custom 'prefix'
filehandler = logging.FileHandler(
    os.path.join("../logs", args.main_folder, args.sub_folder, args.prefix + "_log.txt")
)
filehandler.setLevel(logging.INFO)  # Set the level for file logging to INFO

# Create a StreamHandler to log the output to the console (stdout)
streamhandler = logging.StreamHandler()
streamhandler.setLevel(logging.INFO)  # Set the level for console logging to INFO

# Create a logger object
logger = logging.getLogger("")  # Create or get the root logger
logger.setLevel(
    logging.INFO
)  # Set the global logging level to INFO (this applies to both handlers)

# Disable matplotlib font manager warnings
logging.getLogger("matplotlib.font_manager").disabled = True

# Add the FileHandler and StreamHandler to the logger
# The FileHandler writes logs to the file, and the StreamHandler prints them to the console
logger.addHandler(filehandler)
logger.addHandler(streamhandler)

# Set the random seed for NumPy to ensure reproducibility
random_state = 0
np.random.seed(random_state)  # Set the seed for NumPy's random number generator
torch.manual_seed(random_state)  # Set the seed for PyTorch's random number generator

# Set precision for tensor printing in PyTorch
torch.set_printoptions(
    precision=5
)  # This controls how many decimal places to print for PyTorch tensors

# Set device for model (GPU if available, else CPU)
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)  # Automatically choose GPU (cuda) or CPU based on availability

# Enable anomaly detection for debugging gradients
torch.autograd.set_detect_anomaly(
    True
)  # This helps in detecting problematic operations like NaNs or infs in gradients

# Set up TensorBoard logging (useful for visualizing training progress)
writer = SummaryWriter(
    f"../runs/{args.main_folder}/{args.sub_folder}/"
)  # Create a new TensorBoard summary writer for logging

# Initialize the step counter for TensorBoard logging or training steps
step = (
    0  # Keeps track of the training steps (can be used to log metrics in TensorBoard)
)


# Create training dataset
train_dataset = RtmMpasDatasetWholeTimeLarge(
    nc_file=args.train_file,
    root_dir="../",
    from_time=0,
    end_time=1,
    batch_divid_number=int(960 / 6),
    point_folds=1,
    time_folds=1,
    norm_mapping=norm_mapping_fullyear_new,
    point_number=args.train_point_number,
    only_layer=args.only_layer_boolean,
)

# Create testing dataset
test_dataset = RtmMpasDatasetWholeTimeLarge(
    nc_file=args.test_file,
    root_dir="../",
    from_time=0,
    end_time=1,
    batch_divid_number=int(960 / 6),
    point_folds=1,
    time_folds=1,
    norm_mapping=norm_mapping_fullyear_new,
    point_number=args.test_point_number,
    only_layer=args.only_layer_boolean,
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

'''
# ---------------------------------------------
# Dataset Information
# ---------------------------------------------
logger.info(f"Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")

# ---------------------------------------------
# Model Initialization
# ---------------------------------------------
model = load_model(
    model_name=args.model_name, device=device, feature_channel=34, signal_length=57
)

# Log model parameter statistics
model_info = ModelUtils.get_parameter_number(model)
logger.info(f"Model Info: {model_info}")

# Move model to the appropriate device
model = model.to(device)

# ---------------------------------------------
# Loss Functions & Optimizer Setup
# ---------------------------------------------
criterion_mse = nn.MSELoss()
criterion_mae = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.5, patience=5, verbose=True
)

# ---------------------------------------------
# Load Checkpoint (if specified)
# ---------------------------------------------
if args.load_model == "True":
    checkpoint_path = os.path.join(
        "../checkpoints", args.main_folder, args.sub_folder, args.load_checkpoint_name
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
# Output Index Mapping & Training Start
# ---------------------------------------------
index_mapping = {0: "swuflx", 1: "swdflx", 2: "lwuflx", 3: "lwdflx"}
logger.info("Start training...")

previous_time = time.time()

for epoch in range(args.num_epochs):
    loop = tqdm(train_loader)
    model.train()
    sum_train_mse = 0.0
    sum_train_mae = 0.0
    sum_train_swhr = 0.0
    sum_train_lwhr = 0.0
    sum_train_flux_mse = 0.0

    num_samples = 0
    schedule_losses = []
    logger.info(f"epoch:{epoch}, elapse time:{time.time() - previous_time}")
    previous_time = time.time()

    for batch_idx, (feature, targets, auxis) in enumerate(train_loader):
        if epoch == 0 and batch_idx == 0:
            logger.info(
                f"feature shape:{feature.shape}, target shape:{targets.shape}, auxis shape:{auxis.shape}"
            )
        feature_shape = feature.shape
        target_shape = targets.shape
        auxis_shape = auxis.shape

        inner_batch_size = feature_shape[0] * feature_shape[1]
        feature = feature.reshape(
            inner_batch_size, feature_shape[2], feature_shape[3]
        ).to(device=device)
        targets = targets.reshape(
            inner_batch_size, target_shape[2], target_shape[3]
        ).to(device=device)
        auxis = auxis.reshape(inner_batch_size, auxis_shape[2], auxis_shape[3]).to(
            device=device
        )

        predicts = model(feature)

        predicts_unnorm, targets_unnorm = unnormalized_mpas(
            predicts, targets, norm_mapping_fullyear_new, index_mapping
        )
        swhr_predict, swhr_target, lwhr_predict, lwhr_target = get_heat_rate(
            predicts_unnorm, targets_unnorm, auxis
        )

        _, sw_hr_rmse = MSELoss_all(swhr_predict, swhr_target)
        _, lw_hr_rmse = MSELoss_all(lwhr_predict, lwhr_target)

        loss_mse = criterion_mse(predicts[:, 0:4, :], targets[:, 0:4, :])
        loss_mae = criterion_mae(predicts, targets)
        loss_mse_bottom = criterion_mse(predicts[:, :, 0], targets[:, :, 0])
        total_loss = loss_mse + 0.001 * (sw_hr_rmse + lw_hr_rmse)
        flux_mse = criterion_mse(predicts_unnorm[:, 0:2, :], targets_unnorm[:, 0:2, :])

        num_samples = num_samples + feature_shape[0]
        sum_train_mse = sum_train_mse + feature_shape[0] * loss_mse.item()
        sum_train_mae = sum_train_mae + feature_shape[0] * loss_mae.item()
        sum_train_swhr = sum_train_swhr + feature_shape[0] * sw_hr_rmse.item()
        sum_train_lwhr = sum_train_lwhr + feature_shape[0] * lw_hr_rmse.item()
        sum_train_flux_mse = sum_train_flux_mse + feature_shape[0] * flux_mse.item()

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        writer.add_scalar("train_mse", loss_mse.item(), global_step=step)
        writer.add_scalar("train_mae", loss_mae.item(), global_step=step)
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
                    "../checkpoints",
                    args.main_folder,
                    args.sub_folder,
                    args.prefix
                    + str(epoch).zfill(4)
                    + args.save_checkpoint_name
                    + ".pth.tar",
                )

                filename_full = os.path.join(
                    "../checkpoints",
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

    if epoch % 1 == 0:
        train_mse, train_mae, train_swhr, train_lwhr, train_flux_rmse = (
            sum_train_mse / num_samples,
            sum_train_mae / num_samples,
            sum_train_swhr / num_samples,
            sum_train_lwhr / num_samples,
            np.sqrt(sum_train_flux_mse / num_samples),
        )

        target_norm_info = None

        (
            [sw_flux_rmse, lw_flux_rmse, sw_hr_rmse, lw_hr_rmse, sw_hr_mae, lw_hr_mae],
            [
                sw_flux_rmse,
                sw_flux_mbe,
                sw_flux_bottom_rmse,
                sw_flux_bottom_mbe,
                sw_flux_top_rmse,
                sw_flux_top_mbe,
            ],
            [lw_flux_bottom_rmse],
        ) = check_accuracy(
            test_loader,
            model,
            norm_mapping,
            index_mapping,
            device,
            args,
            target_norm_info,
        )

        schedule_losses.append(sw_hr_rmse + lw_hr_rmse)

        logger.info(
            f"epoch: {epoch}, train_mse: {train_mse: .3e}, train_mae: {train_mae: .3e},\
            train_swhr: {train_swhr: .3e}, train_lwhr: {train_lwhr: .3e}, \
            train_flux_rmse:{train_flux_rmse: .3e}"
        )

        logger.info(
            f"sw_flux_rmse:{sw_flux_rmse: .3e}, lw_flux_rmse:{lw_flux_rmse: .3e}, \
            sw_hr_rmse:{sw_hr_rmse: .3e}, lw_hr_rmse:{lw_hr_rmse: .3e}\
            sw_hr_mae:{sw_hr_mae: .3e}, lw_hr_mae:{lw_hr_mae: .3e}"
        )

        logger.info(
            f"sw_flux_rmse:{sw_flux_rmse: .3e}, sw_flux_mbe:{sw_flux_mbe: .3e}, \
            sw_flux_bottom_rmse:{sw_flux_bottom_rmse: .3e}, sw_flux_bottom_mbe:{sw_flux_bottom_mbe: .3e}\
            sw_flux_top_rmse:{sw_flux_top_rmse: .3e}, sw_flux_top_mbe:{sw_flux_top_mbe: .3e}"
        )

        mean_loss = sum(schedule_losses) / len(schedule_losses)
        scheduler.step(mean_loss)
'''
