#!/bin/bash
#SBATCH --job-name=online-grad-select-dpo
#SBATCH --output=./logs/slurm/slurm-%A.%a.out # stdout file
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=tw6664@princeton.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time=12:59:59
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80


# module purge
# module load anaconda3/2020.11
# conda activate /scratch/gpfs/tw6664/safede



python gc_lora_finetune_single_device.py --config configs/llama3/tong_8B_lora_single_device_finali.yaml 
