#!/bin/bash
#SBATCH --job-name=online-grad-select
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=tw8948@princeton.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=192G
#SBATCH --time=11:59:59
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80

#SBATCH --output=/scratch/gpfs/tw8948/slurm-%j.out
#SBATCH --error=/scratch/gpfs/tw8948/slurm-%j.out

DATA_DIR=./data
MODEL_PATH=meta-llama/Llama-2-7b-hf
DATA_SEED=3
JOB_NAME=llama2-7b-p${PERCENTAGE}-lora-seed${DATA_SEED}


method=$1
batch_size=$2
PERCENTAGE=$3 # percentage of the full data to train, you can specify the training file you want to use in the script
NVAL=$4
task=$5
model=$6
lora_alpha=$7
lr=$8
seed=${9:-"42"}
gradient_accumulation_steps=${10:-"1"}
subject=${11:-"world_religions"}


# Set combined_modules based on the task
if [ "$task" = "mmlu" ]; then
    combined_modules="q_proj k_proj v_proj o_proj"  
else
    combined_modules="q_proj k_proj" 
fi


./less/scripts/train/warmup_lora_train.sh "$DATA_DIR" "$MODEL_PATH" "$PERCENTAGE" "$DATA_SEED" "$JOB_NAME" "$method" "$batch_size" "$subject" "$NVAL" "$task" "$combined_modules" "$lora_alpha" "$lr" "$gradient_accumulation_steps" "$seed"

