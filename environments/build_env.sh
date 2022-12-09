source ~/miniconda3/etc/profile.d/conda.sh  # to use conda from .sh script
conda create -y -n chempred_revision
conda activate chempred_revision
conda install -y gensim pandas numpy rdkit
pip install SmilesPE
conda env export > environment.yml