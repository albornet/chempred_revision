#!/bin/sh
#
#SBATCH --partition=private-teodoro-gpu
#SBATCH --time=12:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task 2
#SBATCH --mem-per-cpu=8000
#SBATCH --output=$SLURM_LOG_PATH
#SBATCH --error=$SLURM_ERR_PATH

module load Anaconda3/2022.05
source activate chempred_revision

python open-nmt/train.py --config=$CONFIG_PATH