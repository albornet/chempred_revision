#!/bin/sh
#
for slurm_file_name in $(ls -d $PWD/slurm/data_augment_exp_x1/*)
do
    echo Processing $slurm_file_name
    sbatch $slurm_file_name
done
for slurm_file_name in $(ls -d $PWD/slurm/input_scheme_exp/*)
do
    echo Processing $slurm_file_name
    sbatch $slurm_file_name
done
sleep 3
sacct