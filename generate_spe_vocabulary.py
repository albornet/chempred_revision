import os
import codecs
import pandas as pd
import selfies as sf
from SmilesPE.learner import learn_SPE


INSOURCE = 'uspto'  # 'uspto', 'zink'
CODE_PATH_SMILES = os.path.join('data', 'original', 'spe_codes_smiles.txt')
CODE_PATH_SELFIES = os.path.join('data', 'original', 'spe_codes_selfies.txt')


def main():
    print('Building spe vocabulary (smiles pair encoding)')
    build_spe_vocab('smiles')
    print('Building spe vocabulary (selfies pair encoding)')
    build_spe_vocab('selfies')


def build_spe_vocab(informat):
    # Define loaded molecules where the vocabulary will be saved
    infile = load_smiles_molecules()
    outpath = {'smiles': CODE_PATH_SMILES,
               'selfies': CODE_PATH_SELFIES}[informat]
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
              augmentation=0,  # 1 -> 0 for selfies! so, 0 also for smiles
              verbose=True,
              total_symbols=True)


def smiles_to_selfies(smiles):
    try:
        return sf.encoder(smiles)
    except:
        return None


def load_smiles_molecules():
    if INSOURCE == 'zink':
        inpath = os.path.join('data', 'zink', 'all_smiles_filtered.txt')
        infile = pd.read_csv(inpath).smi.tolist()
        return [smiles for smiles in infile if '@' not in smiles]  # no 3D 
    elif INSOURCE == 'uspto':
        original_dir = os.path.join('data', 'original')
        all_molecules = []
        with open(os.path.join(original_dir, 'src-train.txt'), 'r') as src_in,\
            open(os.path.join(original_dir, 'tgt-train.txt'), 'r') as tgt_in:
            for src, tgt in zip(src_in.readlines(), tgt_in.readlines()):
                all_molecules.extend(parse_rxn(src, tgt))
        return [m for m in all_molecules if len(m) > 0]  # + '.'
    else:
        raise ValueError('Invalide input source (zink, uspto)')


def parse_rxn(src, tgt):
    src, tgt = src.strip(), tgt.strip()
    if '  >' in src: src = src.replace('  >', ' > ')
    precursors = src.split(' > ')
    reactants, reagents = [p.split(' . ') for p in precursors]
    products = tgt.split(' . ')  # should we pick only the largest?
    rxn_molecules = reactants + reagents + products
    return [m.replace(' ', '') for m in rxn_molecules]


if __name__ == '__main__':
    main()
