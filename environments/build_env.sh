source ~/miniconda3/etc/profile.d/conda.sh  # to use conda from .sh script
conda create -y -n chempred_revision
conda activate chempred_revision
conda install -y gensim pandas numpy tqdm
conda install -y selfies rdkit -c conda-forge
pip install SmilesPE
conda env export > environment.yml