#!/bin/bash

# Base configuration file
BASEFILE="configs/llama3/tong_8B_lora_single_device_newi.yaml"

# Command line arguments
method=$1
lora_rank=$2
lora_alpha=$3
batch_size=$4
n_val=$5
fracinv=$6
lr=$7

# Generate the output file name according to the Python script's naming scheme
OUTPUT_FILE="${BASEFILE%.yaml}_Exp_${method}_LR${lr}_LoRA_R${lora_rank}_alpha${lora_alpha}_bs${batch_size}_nval${n_val}_fracinv${fracinv}.yaml"

# Call the Python script to update the YAML file
python update_yaml.py --input_file "$BASEFILE" --method "$method" --lora_rank "$lora_rank" --lora_alpha "$lora_alpha" --batch_size "$batch_size" --n_val "$n_val" --fracinv "$fracinv" --lr "$lr"

# Use the generated YAML configuration file to run the fine-tuning script
python -u gc_lora_finetune_single_device.py --config "$OUTPUT_FILE"

# sh run_gc.sh Regular  8 0.1 16 80 2.0 1e-05