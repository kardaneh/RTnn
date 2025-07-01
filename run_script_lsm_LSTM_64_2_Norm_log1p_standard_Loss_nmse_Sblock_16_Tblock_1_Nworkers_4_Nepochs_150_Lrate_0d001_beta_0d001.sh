#!/bin/bash
echo "Time = 20250701_085917"
echo "Running training with model: LSTM_64_2, loss: nmse"

python src/main_lsm.py \
    --root_dir "./" \
    --main_folder "Prod__LSTM_64_2_Sblock_16_Tblock_1_Nworkers_4_Nepochs_150" \
    --sub_folder "Norm_log1p_standard_Lrate_0d001_beta_0d001_Loss_nmse_date_20250701_085917" \
    --train_data_files "/leonardo_work/EUHPC_D17_070/Data_LSM_1990-2000/" \
    --test_data_files "/leonardo_work/EUHPC_D17_070/Data_LSM_1990-2000/" \
    --train_year "1998" \
    --test_year "1999" \
    --prefix "Norm_log1p_standard_Lrate_0d001_beta_0d001_Loss_nmse_date_20250701_085917" \
    --dataset_type "LSM" \
    --loss_type "nmse" \
    --learning_rate "0.001" \
    --beta "0.001" \
    --batch_size "16" \
    --tbatch "1" \
    --model_name "LSTM_64_2" \
    --num_workers "4" \
    --num_epochs "150" \
    --save_mode "True" \
    --save_checkpoint_name "model" \
    --save_per_samples 10000 \
    --load_model "False"
