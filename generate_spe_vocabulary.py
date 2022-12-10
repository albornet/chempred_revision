import os
import codecs
import pandas as pd
import selfies as sf
from SmilesPE.learner import learn_SPE


def main():
    inpath = os.path.join('data', 'zink', 'all_smiles_filtered.txt')
    infile = pd.read_csv(inpath).smi.tolist()
    print('Building spe vocabulary (smiles pair encoding)')
    build_spe_vocab(infile, 'smiles')
    print('Building spe vocabulary (selfies pair encoding)')
    build_spe_vocab(infile, 'selfies')


def build_spe_vocab(infile, informat):
    # Define where the vocabulary will be saved
    outpath = os.path.join('data', 'zink', 'spe_vocab_%s.txt' % informat)
    outfile = codecs.open(outpath, 'w')
    
    # Change smiles molecules to selfies if needed
    if informat == 'selfies':
        infile = [smiles_to_selfies(smiles) for smiles in infile]
        infile = [selfies for selfies in infile if selfies is not None]
    
    # Train spe (smiles-bpe) tokenizer on the zink data set
    learn_SPE(infile=infile,
              outfile=outfile,
              num_symbols=30000,
              min_frequency=2000,
              augmentation=0,  # 1 -> 0 for selfies! so keep 0 also for smiles
              verbose=True,
              total_symbols=True)


def smiles_to_selfies(smiles):
    try:
        return sf.encoder(smiles)
    except:
        return None


if __name__ == '__main__':
    main()
