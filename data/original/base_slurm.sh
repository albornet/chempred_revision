#!/bin/sh
#
#SBATCH --partition=shared-gpu
#SBATCH --time=02:15:00
#SBATCH --gpus=ampere:1
#SBATCH --cpus-per-task 2
#SBATCH --mem-per-cpu=8000
#SBATCH --output=./logs/bert_M.log
#SBATCH --error=./logs/bert_M.err

module load Anaconda3/2020.07
source activate torch-env

python open-nmt/train.py --config=$CONFIG_PATH