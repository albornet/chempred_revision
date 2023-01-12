#!/bin/sh

for slurm_file_name in $(ls -d $PWD/slurm/train/input_scheme_exp/*); do
    echo Processing $slurm_file_name
    sbatch $slurm_file_name
done

for slurm_file_name in $(ls -d $PWD/slurm/train/data_augment_exp_x1/*); do
    echo Processing $slurm_file_name
    sbatch $slurm_file_name
done

for slurm_file_name in $(ls -d $PWD/slurm/train/data_augment_exp_x2/*); do
    echo Processing $slurm_file_name
    sbatch $slurm_file_name
done

for slurm_file_name in $(ls -d $PWD/slurm/train/data_augment_exp_x5/*); do
    echo Processing $slurm_file_name
    sbatch $slurm_file_name
done

for slurm_file_name in $(ls -d $PWD/slurm/train/data_augment_exp_x10/*); do
    echo Processing $slurm_file_name
    sbatch $slurm_file_name
done

for slurm_file_name in $(ls -d $PWD/slurm/train/data_augment_exp_x20/*); do
    echo Processing $slurm_file_name
    sbatch $slurm_file_name
done

sleep 5
sacct