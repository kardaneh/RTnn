#!/bin/bash
#SBATCH -A EUHPC_D17_070
#SBATCH -p boost_usr_prod
#SBATCH --time=00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=32
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-gpu=64G
#SBATCH --job-name=lsm_LSTM_64_2_Norm_log1p_standard_Loss_nmse_Sblock_32_Tblock_1_Nworkers_4_Nepochs_150_Lrate_0d001_beta_0d001
#SBATCH --error=lsm_training_LSTM_64_2_Norm_log1p_standard_Loss_nmse_Sblock_32_Tblock_1_Nworkers_4_Nepochs_150_Lrate_0d001_beta_0d001.err
#SBATCH --output=lsm_training_LSTM_64_2_Norm_log1p_standard_Loss_nmse_Sblock_32_Tblock_1_Nworkers_4_Nepochs_150_Lrate_0d001_beta_0d001.out

module load profile/deeplrn
module load cineca-ai
source /leonardo/home/userexternal/kardaneh/RTnn/python-venv/bin/activate

./run_script_lsm_LSTM_64_2_Norm_log1p_standard_Loss_nmse_Sblock_32_Tblock_1_Nworkers_4_Nepochs_150_Lrate_0d001_beta_0d001.sh
