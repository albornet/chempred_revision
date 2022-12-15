read -n 1 -p "Have you activated the correct conda environment? (y/n) " confirm
if [ $confirm == "y" ]; then
    echo -e "\nInstalling environment..."
    eval "$(conda shell.bash hook)"
    conda install -y gensim pandas numpy tqdm
    conda install -y selfies rdkit -c conda-forge
    pip install SmilesPE
    cd open-nmt && pip install -e . && cd ..
    conda env export > environments/environment.yml
    echo "Environment installed successfully!"
else
    if [ $confirm == "n" ]; then
        echo -e "\nNot installing environment and quitting."
    else
        echo -e "\nBad input. Not installing environment and quitting."
    fi
fi
