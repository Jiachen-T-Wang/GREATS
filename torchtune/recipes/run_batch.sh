#!/bin/bash
#SBATCH --job-name=online-grad-select-torchtune
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=tw8948@princeton.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=192G
#SBATCH --time=23:59:59
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80

#SBATCH --output=/scratch/gpfs/tw8948/slurm-%j.out
#SBATCH --error=/scratch/gpfs/tw8948/slurm-%j.out



method=$1
bs=$2

sh run_gc.sh $method  8 0.1 $bs 80 2.0 1e-05



# # Command line arguments
# method=$1
# lora_rank=$2
# lora_alpha=$3
# batch_size=$4
# n_val=$5
# fracinv=$6
# lr=$7
