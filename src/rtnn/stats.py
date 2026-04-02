import os
import glob
import logging
import numpy as np
from rtnn.file_helper import FileUtils
from rtnn.plot_helper import stats

FileUtils.makedir(os.path.join("stats", "."))
train_sbatch_files = np.sort(
    glob.glob(
        "/leonardo_work/EUHPC_D17_070/Data_LSM_1990-2000/" + f"rtnetcdf_*_{1998}.nc"
    )
)[::]

log_dir = os.path.join("stats", ".")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "stat_log.txt")

logger = logging.getLogger("")
logger.setLevel(logging.INFO)

if logger.hasHandlers():
    logger.handlers.clear()

file_handler = logging.FileHandler(log_file, mode="w")
file_handler.setLevel(logging.INFO)

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)

norm_mapping = stats([train_sbatch_files[0]], logger, os.path.join("stats", "."))
for var_name, stats_dict in norm_mapping.items():
    print(f"Variable: {var_name}")
    for stat_key, value in stats_dict.items():
        print(f"  {stat_key}: {value:.4e}")
