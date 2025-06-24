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

# Create training dataset
train_dataset = DataPreprocessor(
        logger = logger,
        dfs = train_sbatch_files,
        sbatch=len(train_sbatch_files),
        stime=0, 
        etime=train_df.sizes['time'],
        tbatch=args.tbatch,
        norm_mapping=norm_mapping
        )

test_dataset = DataPreprocessor(
        logger = logger,
        dfs = test_sbatch_files,
        sbatch=len(test_sbatch_files),
        stime=0,
        etime=train_df.sizes['time'],
        tbatch=24,
        norm_mapping=norm_mapping
        )

# Create DataLoader for training
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.num_workers,
    pin_memory=True)

# Create DataLoader for testing
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.num_workers,
    pin_memory=True)

# ---------------------------------------------
# Dataset Information
# ---------------------------------------------
logger.info(f"Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")

# ---------------------------------------------
# Model Initialization
# ---------------------------------------------
model = load_model(model_name=args.model_name, device=device, feature_channel=6, signal_length=10)

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
logger.info("Start training...")

metric_suffix = args.loss_type.lower()
assert metric_suffix in ['mse', 'mae', 'nmae', 'nmse'], \
    "Invalid loss_type (should be one of 'mse', 'mae', 'nmae', 'nmse')"

if metric_suffix == "mse":
    func = mse_all
elif metric_suffix == "mae":
    func = mae_all
elif metric_suffix == "nmae":
    func = nmae_all
elif metric_suffix == "nmse":
    func = nmse_all
else:
    raise ValueError(f"Unsupported loss type: {metric_suffix}")

main_key = f"{metric_suffix}"
abs12_key = f"abs12_{metric_suffix}"
abs34_key = f"abs34_{metric_suffix}"

train_metrics = {
    main_key: MetricTracker(),
    abs12_key: MetricTracker(),
    abs34_key: MetricTracker()
    }

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

    loop = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}")
    for batch_idx, (feature, targets) in loop:
        #logger.info(f"batch idx:{batch_idx}")
        if epoch == 0 and batch_idx == 0:
            logger.info(f"feature shape:{feature.shape}, target shape:{targets.shape}")

        feature_shape = feature.shape
        target_shape = targets.shape

        inner_batch_size = feature_shape[0] * feature_shape[1]
        feature = feature.reshape(inner_batch_size, feature_shape[2], feature_shape[3]).to(device=device)
        targets = targets.reshape(inner_batch_size, target_shape[2], target_shape[3]).to(device=device)

        predicts = model(feature)

        predicts_unnorm, targets_unnorm = unnorm_mpas(predicts, targets, norm_mapping, index_mapping)
        abs12_predict, abs12_target, abs34_predict, abs34_target = calc_abs(predicts_unnorm, targets_unnorm)
        #abs12_predict, abs12_target, abs34_predict, abs34_target = calc_abs(predicts, targets)
        
        metric_values = {
                main_key: func(predicts_unnorm, targets_unnorm),
                abs12_key: func(abs12_predict, abs12_target),
                abs34_key: func(abs34_predict, abs34_target)
                }

        main_count, main_val = metric_values[main_key]
        abs12_count, abs12_val = metric_values[abs12_key]
        abs34_count, abs34_val = metric_values[abs34_key]

        weighted_loss = (1.0 - args.beta) * main_val * main_count + args.beta * (abs12_val * abs12_count + abs34_val * abs34_count)
        total_count = (1.0 - args.beta) * main_count + args.beta * (abs12_count + abs34_count)
        total_loss = weighted_loss / total_count

        loop.set_postfix(loss=total_loss.item())
        
        for key, (count, value) in metric_values.items():
            train_metrics[key].update(value.item(), count)

        train_loss.update(total_loss.item(),  1)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        writer.add_scalar(f"train_{main_key}", metric_values[main_key][1].item(), global_step=step)
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
    #for key, tracker in train_metrics.items():
    #    logger.info(f"{key} -> count: {tracker.count}, value: {tracker.value}")
    train_loss_history[epoch] = train_loss.getmean()
    if epoch % 1 == 0:

        valid_loss, valid_metrics = check_accuracy_evaluate_lsm(
                test_loader,
                model,
                norm_mapping,
                index_mapping,
                device,
                args,
                epoch)

        valid_loss_history[epoch] = valid_loss
        logger.info(f"train_loss: {train_loss_history[epoch]:.3e} | valid_loss: {valid_loss_history[epoch]:.3e}")
        for key, meter in train_metrics.items():
            train_value = meter.getsqrtmean() if key.endswith('mse') else meter.getmean()
            train_metrics_history[key].append(train_value)

            assert key in valid_metrics, f"Missing key '{key}' in valid_metrics"

            valid_value = valid_metrics[key]
            valid_metrics_history[key].append(valid_value)

            logger.info(f"train_{key}: {train_value:.3e} | valid_{key}: {valid_value:.3e}")

        logger.info("")
        schedule_losses.append(valid_metrics[main_key])
        mean_loss = sum(schedule_losses) / len(schedule_losses)
        scheduler.step(mean_loss)

base_dir = os.path.join("results", args.main_folder, args.sub_folder)
plot_metric_histories(train_metrics_history, valid_metrics_history, os.path.join(base_dir, f"metrics_panel.png"))
plot_loss_histories(train_loss_history, valid_loss_history, os.path.join(base_dir, f"training_validation_loss.png"))
