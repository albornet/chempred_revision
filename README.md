
Code for Jaume-Santero-2022 revision
======================
This repository contains the code that was written to implement the revisions asked by the reviewers for the paper *Transformer performance for chemical reactions: analysis of different predictive and evaluation scenarios*

How to reproduce the reported results
-------------------------------------
* Create an environment with the necessary packages (here with conda)
```
$ conda env create --name chempred_revision --file environments/environment.yml
$ conda activate chempred_revision
```

* Generate all required datasets and pre-trained embedding vectors
```
$ python generate_all_datasets.py
```

* Train all models (long step! here are two different ways)
```
$ FOR BOTH WAYS
$ python write_train_configs.py  # write configuration files for training
$
$ # THN, EITHER RUN IT ON YOUR OWN COMPUTER / SERVER (WITH AT LEAST 1 GPU)
$ python write_vocab_and_slurm_files.py -v  # vocabulary generation
$ python train_all_models.py  # all training scripts run one by one
$
$ # OR RUN IT ON AN HPC CLUSTER
$ # You should adapt ./data/original/base_slurm.sh to your needs
$ # You should install a conda environment named "chempred_revision" on your personal HPC space, where open-nmt is installed with "pip install -e ." from ./open-nmt)
$ python write_vocab_and_slurm_files.py -v -s  # vocabulary and slurm script generation
$ sbatch slurm/<path_to_slurm_file>  # do this for all slurm files located in ./slurm (they are grouped by runtime)
```

* Generate predictions using the test data, with all trained model, and write them to text files
```
$ python write_test_and_roundtrip_configs.py  # write configuration files for testing
$ python test_and_roundtrip_all_models.py  # generate predictions for all models
```

* Evaluate the performance of all models (top-k and roundtrip) and generate all figures
```
$ compute_topk_and_roundtrip_accuracy.py
$ create_figures_x_and_y.py  # coming soon
```
