#!/bin/bash
echo "Time = 20250625_161601"
echo "Running training with model: LSTM_96_2, loss: nmse"

python src/main_lsm.py \
    --root_dir "./" \
    --main_folder "Prod__LSTM_96_2_bs_16_tb_1_nw_16_ep_120" \
    --sub_folder "lr_0d001_beta_0d001_lf_nmse_date_20250625_161601" \
    --train_data_files "/leonardo_work/EUHPC_D17_070/Data_LSM_1990-2000/" \
    --test_data_files "/leonardo_work/EUHPC_D17_070/Data_LSM_1990-2000/" \
    --train_year "1998" \
    --test_year "1999" \
    --prefix "lr_0d001_beta_0d001_lf_nmse_date_20250625_161601" \
    --dataset_type "LSM" \
    --loss_type "nmse" \
    --learning_rate "0.001" \
    --beta "0.001" \
    --batch_size "16" \
    --tbatch "1" \
    --model_name "LSTM_96_2" \
    --num_workers "16" \
    --num_epochs "120" \
    --save_mode "True" \
    --save_checkpoint_name "model" \
    --save_per_samples 10000 \
    --load_model "False"
