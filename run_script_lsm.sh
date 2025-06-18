#!/bin/bash
DATE=`date +"%Y%m%d"`
TODAY=`date +"%Y%m%d_%H%M%S"`
echo "Time = $TODAY"
model_name="LSTM"
batch_size=12
num_workers=16
num_epochs=30
learning_rate=0.001
train_data_files="/leonardo_work/EUHPC_D17_070/Data_LSM_1990-2000/"
test_data_files="/leonardo_work/EUHPC_D17_070/Data_LSM_1990-2000/"
main_folder="LSM"
learning_rate_str="${learning_rate//./d}"
sub_folder="${model_name}_bs${batch_size}_nw${num_workers}_ep${num_epochs}_lr${learning_rate_str}_date${TODAY}"
prefix="$sub_folder"

echo "Running training with:"
echo "  Model Name      : $model_name"
echo "  Sub Folder      : $sub_folder"
echo "  Prefix          : $prefix"
echo "  Batch Size      : $batch_size"
echo "  Num Workers     : $num_workers"
echo "  Num Epochs      : $num_epochs"
echo "  Learning Rate   : $learning_rate"
echo "  Training Data Files Path : $train_data_files"
echo "  Testing Data Files Path : $test_data_files"
echo ""

python src/main_lsm.py \
    --root_dir "./" \
    --main_folder "$main_folder" \
    --sub_folder "$sub_folder" \
    --train_data_files "$train_data_files" \
    --test_data_files "$test_data_files" \
    --prefix "$prefix" \
    --dataset_type "LSM" \
    --loss_type "v01" \
    --learning_rate "$learning_rate" \
    --batch_size "$batch_size" \
    --model_name "$model_name" \
    --num_workers "$num_workers" \
    --num_epochs "$num_epochs" \
    --save_mode "True" \
    --save_checkpoint_name "model" \
    --save_per_samples 10000 \
    --load_model "False"
