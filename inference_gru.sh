#!/bin/bash
DATE=`date +"%Y%m%d"`
TODAY=`date +"%Y%m%d_%H%M%S"`
echo "Time = $TODAY"
model="Prod"
model_name="GRU_64_2"
loss_type="nmse"
batch_size=16
time_batch=1
num_workers=16
num_epochs=150
learning_rate=0.001
beta=0.001
train_data_files="/leonardo_work/EUHPC_D17_070/Data_LSM_1990-2000/"
test_data_files="/leonardo_work/EUHPC_D17_070/Data_LSM_1990-2000/"
train_year="1998"
test_year="2000"
lr_str=${learning_rate//./d}
beta_str=${beta//./d}
main_folder="Prod__GRU_64_2_bs_16_tb_2_nw_16_ep_150"
sub_folder="lr_0d001_beta_0d001_lf_nmse_date_20250625_025123"
prefix="$sub_folder"

echo "Running training with:"
echo "  Model Name      : $model_name"
echo "  Sub Folder      : $sub_folder"
echo "  Prefix          : $prefix"
echo "  Batch Size      : $batch_size"
echo "  Temporal Batch Size      : $time_batch"
echo "  Num Workers     : $num_workers"
echo "  Num Epochs      : $num_epochs"
echo "  Learning Rate   : $learning_rate"
echo "  Loss Type       : $loss_type"
echo "  Weighting factor   : $beta"
echo "  Training Data Files Path : $train_data_files"
echo "  Testing Data Files Path : $test_data_files"
echo "  Train Year       : $train_year"
echo "  Test Year        : $test_year"
echo ""

python src/inference_lsm.py \
    --root_dir "./" \
    --main_folder "$main_folder" \
    --sub_folder "$sub_folder" \
    --train_data_files "$train_data_files" \
    --test_data_files "$test_data_files" \
    --train_year "$train_year" \
    --test_year "$test_year" \
    --prefix "$prefix" \
    --dataset_type "LSM" \
    --loss_type "$loss_type" \
    --learning_rate "$learning_rate" \
    --beta "$beta" \
    --batch_size "$batch_size" \
    --tbatch "$time_batch" \
    --model_name "$model_name" \
    --num_workers "$num_workers" \
    --num_epochs "$num_epochs" \
    --save_mode "True" \
    --save_checkpoint_name "model" \
    --save_per_samples 10000 \
    --load_model "True"\
    --load_checkpoint_name "lr_0d001_beta_0d001_lf_nmse_date_20250625_0251230147model.pth.tar"

