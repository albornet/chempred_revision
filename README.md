
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

* Generate the configuration files used to train all models with open-nmt
```
$ python generate_train_configs.py
```

* Train all models (this takes a while, you might consider splitting the jobs on multiple GPUs!)
```
$ python train_all_models.py
```

* Generate the configuration files used to test all models and generate roundtrip datasets
```
$ python generate_test_and_roundtrip_configs.py
```

* Generate predictions using the test data, with all trained model, and write them to text files
```
$ python test_and_roundtrip_all_models.py
```

* Evaluate the performance of all models (top-k and roundtrip)
```
$ compute_topk_and_roundtrip_accuracy.py
```

* Generate figures with the results
```
$ create_figures_x_and_y.py  # coming soon
```

