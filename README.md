
Code for Jaume-Santero-2022 revision
======================
This repository contains the code that was written to implement the revisions asked by the reviewers for the paper *Transformer performance for chemical reactions: analysis of different predictive and evaluation scenarios*

How to reproduce the reported results
-------------------------------------
* Generate all required datasets and embedding vectors
```
$ conda env create --name chempred_revision --file environments/environment.yml
$ conda activate chempred_revision
```

* Generate all required datasets and embedding vectors
```
$ python generate_all_datasets.py
```

* Generate the configuration files used to run all simulations with open-nmt
```
$ python generate_all_omt_configs.py
```

* Train all models (this might take a while, approximately 200 days on a single GPU!)
```
$ python train_all_models.py
```

* Generate predictions using the test data, with all trained model, and write them to text files
```
$ python test_all_models.py
```

* Generate all figures with the results
```
$ Yet to come!
```
