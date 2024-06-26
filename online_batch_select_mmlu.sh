#!/bin/bash
#SBATCH --job-name=online-grad-select-LESS
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=tw8948@princeton.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=192G
#SBATCH --time=5:59:59
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80

#SBATCH --output=/scratch/gpfs/tw8948/slurm_output/slurm-%j.out
#SBATCH --error=/scratch/gpfs/tw8948/slurm_output/slurm-%j.err

DATA_DIR=./data
MODEL_PATH=meta-llama/Llama-2-7b-hf
DATA_SEED=3
JOB_NAME=llama2-7b-p${PERCENTAGE}-lora-seed${DATA_SEED}

method=$1
batch_size=$2
subject=$3
PERCENTAGE=$4 # percentage of the full data to train, you can specify the training file you want to use in the script
NVAL=$5

./less/scripts/train/warmup_lora_train.sh "$DATA_DIR" "$MODEL_PATH" "$PERCENTAGE" "$DATA_SEED" "$JOB_NAME" "$method" "$batch_size" "$subject" "$NVAL"
