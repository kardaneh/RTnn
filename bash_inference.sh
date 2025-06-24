#!/bin/bash
######################
#SBATCH -A EUHPC_D17_070
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time 00:30:00
#SBATCH -N 1
#SBATCH --ntasks=32
#SBATCH --gpus=2
#SBATCH --mem=64G
#SBATCH --job-name=lsm_traning
#SBATCH --error=lsm_training_%j.err
#SBATCH --output=lsm_training_%j.out
module load profile/deeplrn
module load cineca-ai
source /leonardo/home/userexternal/kardaneh/RTnn/python-venv/bin/activate
#====================================================
# Access to module command
#====================================================
./run_script_inference_lsm.sh
