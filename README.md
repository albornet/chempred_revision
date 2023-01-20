Code for Jaume-Santero-2022 revised manuscript
======================
This repository contains the code that was written to generate the results in the manuscript *Transformer performance for chemical reactions: analysis of different predictive and evaluation scenarios*, including additional simulations asked by the reviewers.

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
# EITHER USE ON YOUR OWN COMPUTER / SERVER (WITH AT LEAST 1 GPU)
$ python write_train_configs.py  # write configuration files for training
$ python write_vocab_and_slurm_files.py -v  # vocabulary generation only
$ python train_all_models.py  # all training scripts run one by one

# OR USE AN HPC CLUSTER
# -> You should adapt ./data/original/base_slurm.sh to your needs
# -> You may also need to update write_vocab_and_slurm_files.py to your needs
# -> You should install a conda environment named "chempred_revision" on your personal HPC space,
#    where open-nmt is installed with "pip install -e ." from ./open-nmt)
$ python write_train_configs.py  # write configuration files for training
$ python write_vocab_and_slurm_files.py -v -s  # vocabulary and slurm script generation
$ ./sbatch_all_models.sh  # all training scripts run as soon as there is an available GPU
```

* Generate predictions using the test data, with all trained model, and write them to text files (again, two ways)
```
# EITHER USE YOUR OWN COMPUTER / SERVER (WITH AT LEAST 1 GPU)
$ python write_test_and_roundtrip_configs.py -t  # write configuration files for testing tasks
$ python test_and_roundtrip_all_models.py -t # generate test predictions for all models
$ python write_test_and_roundtrip_configs.py -r  # write configuration files for roundtrip tasks, using test outputs
$ python test_and_roundtrip_all_models.py -r # generate roundtrip predictions for all reactant prediction models

# OR USE AN HPC CLUSTER
$ python write_test_and_roundtrip_configs.py -t  # write configuration files for testing tasks
$ ./sbatch_test_all_models.sh # generate test predictions for all models
$ python write_test_and_roundtrip_configs.py -r  # write configuration files for roundtrip tasks, using test outputs
$ ./sbatch_roundtrip_all_models.sh # generate roundtrip predictions for all reactant prediction models
```

* Evaluate the performance of all models (top-k and roundtrip accuracies) and generate all figures
```
$ python evaluate_all_models.py
$ python plot_all_result_figures.py
```
