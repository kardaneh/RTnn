import argparse
import os, glob
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
from data_helper_lsm import DataPreprocessor
from model_prepare import load_model
from model_helper import ModelUtils
from evaluate_helper import (
    unnorm_mpas,
    calc_abs,
    check_accuracy_evaluate_lsm,
    MetricTracker, mse_all, mae_all, nmae_all, nmse_all
    )
from plot_helper import plot_metric_histories, plot_loss_histories

def parse_args():
    parser = argparse.ArgumentParser(description="Train the RTM model")
    parser.add_argument("--root_dir", type=str, default="", help="Root directory")
    parser.add_argument("--train_data_files", type=str, default="", help="Path to training dataset")
    parser.add_argument("--test_data_files", type=str, default="", help="Path to training dataset")
    parser.add_argument("--train_year", type=str, default="1998", help="Year for training data")
    parser.add_argument("--test_year", type=str, default="2000", help="Year for testing data")
    parser.add_argument("--main_folder", type=str, default="temp", help="Main folder name")
    parser.add_argument("--sub_folder", type=str, default="temp", help="Sub-folder name")
    parser.add_argument("--prefix", type=str, default="temp", help="Prefix for saving")
    parser.add_argument("--dataset_type", type=str, default="Large", help="Type of dataset")
    parser.add_argument(
    '--loss_type',
    type=str,
    default='mse',
    choices=['mse', 'mae', 'nmae', 'nmse'],
    help='Loss type to use for flux weighting (mse, mae, nmae, or nmse)')
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--beta", type=float, default=0.05, help="Weighting factor between RMSE and abs loss terms")
    parser.add_argument("--batch_size", type=int, default=200, help="Batch size")
    parser.add_argument("--tbatch", type=int, default=24, help="Time batch length for the DataPreprocessor")
    parser.add_argument("--model_name", type=str, default="FC", help="Model name")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--save_model", choices=("True", "False"), default="False", help="Save the trained model")
    parser.add_argument("--save_checkpoint_name",type=str,default="test.pth.tar",help="Checkpoint file name")
    parser.add_argument("--save_per_samples",type=int,default=10000,help="Frequency of saving checkpoints")
    parser.add_argument("--load_model",choices=("True", "False"),default="False",help="Load a pre-trained model")
    parser.add_argument("--load_checkpoint_name",type=str,default="test.pth.tar",help="Checkpoint file to load")
    parser.add_argument("--random_throw",choices=("True", "False"),default="False",help="Random throw option")
    parser.add_argument("--only_layer",choices=("True", "False"),default="False",help="Use only a specific layer")
    args = parser.parse_args()
    return args


args = parse_args()
train_sbatch_files = np.sort(glob.glob(args.train_data_files + f"rtnetcdf_*_{args.train_year}.nc"))[::]
test_sbatch_files = np.sort(glob.glob(args.test_data_files + f"rtnetcdf_*_{args.test_year}.nc"))[::]
train_df = xr.open_dataset(train_sbatch_files[0], engine="netcdf4")

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
log_file = os.path.join(log_dir, f"{args.prefix}_log.txt")

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
index_mapping = {0: "collim_alb", 1: "collim_tran", 2: "isotrop_alb", 3: "isotrop_tran"}
step = (0)
logger.info(f"NetCDF files path: {args.train_data_files}")
logger.info(f"NetCDF files path: {args.test_data_files}")
logger.info(f"Found {len(train_sbatch_files)} training files:")
for f in train_sbatch_files:
    logger.info(f"  {f}")
logger.info(f"Found {len(test_sbatch_files)} test files:")
for f in test_sbatch_files:
    logger.info(f"  {f}")

norm_mapping = {
        var: {
            'mean': train_df[var].mean().item(),
            'std': train_df[var].std().item()
            }
        for var in train_df.data_vars
        }

test_dataset = DataPreprocessor(
        logger = logger,
        dfs = test_sbatch_files,
        sbatch=len(test_sbatch_files),
        stime=0,
        etime=train_df.sizes['time'],
        tbatch=args.tbatch,
        norm_mapping=norm_mapping
        )

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.num_workers,
    pin_memory=True)

# ---------------------------------------------
# Model Initialization
# ---------------------------------------------
model = load_model(model_name=args.model_name, device=device, feature_channel=6, signal_length=10)

model_info = ModelUtils.get_parameter_number(model)
logger.info(f"Model Info: {model_info}")

model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

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
logger.info("Start inference...")
valid_loss, valid_metrics = check_accuracy_evaluate_lsm(
        test_loader,
        model,
        norm_mapping,
        index_mapping,
        device,
        args,
        args.num_epochs - 1)

logger.info(f"valid_loss: {valid_loss:.3e}")
for key, meter in valid_metrics.items():
    valid_value = valid_metrics[key]
    logger.info(f"valid_{key}: {valid_value:.3e}")
    logger.info("")
