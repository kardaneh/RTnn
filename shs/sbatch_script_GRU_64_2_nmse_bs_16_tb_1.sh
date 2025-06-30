#!/bin/bash
#SBATCH -A EUHPC_D17_070
#SBATCH -p boost_usr_prod
#SBATCH --time=15:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=32
#SBATCH --gpus-per-node=2
#SBATCH --mem-per-gpu=64G
#SBATCH --job-name=lsm_GRU_64_2_nmse_bs_16_tb_1
#SBATCH --error=lsm_training_GRU_64_2_nmse_bs_16_tb_1_%j.err
#SBATCH --output=lsm_training_GRU_64_2_nmse_bs_16_tb_1_%j.out

module load profile/deeplrn
module load cineca-ai
source /leonardo/home/userexternal/kardaneh/RTnn/python-venv/bin/activate

./run_script_lsm_GRU_64_2_nmse_bs_16_tb_1.sh
