import os
import re
import random
import selfies as sf
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
from tqdm import tqdm
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from SmilesPE.tokenizer import SPE_Tokenizer
from gensim.models import Word2Vec


TASKS = [
    'product-pred',
    'product-pred-noreag',
    'reactant-pred',
    'reactant-pred-noreag',
    'reactant-pred-single',
    'reagent-pred'
]
USPTO_50K_TASKS = [t for t in TASKS if t != 'reagent-pred']
FOLDS = [1, 2, 5, 10, 20]
SPLITS = ['test', 'val', 'train']
DATA_DIR = os.path.abspath('data')
ORIGINAL_DIR = os.path.join(DATA_DIR, 'original')
SPE_ENCODER_PATH_SMILES = os.path.join(ORIGINAL_DIR, 'spe_codes_smiles.txt')
SPE_ENCODER_PATH_SELFIES = os.path.join(ORIGINAL_DIR, 'spe_codes_selfies.txt')


def main():
    random.seed(1234)  # this is enough for replicable augmentation
    create_smiles_datasets()  # original data -> all tasks & data augmentations
    create_selfies_datasets()  # smiles data -> selfies data
    create_spe_datasets()  # smiles and selfies atom data -> spe data
    create_w2v_embeddings()  # smiles, selfies, atom, bpe -> added w2v vectors


def create_smiles_datasets():
    print('\nStarted generating smiles datasets')
    for task in TASKS:
        print('-- Task: %s' % task)
        for fold in FOLDS:
            print('---- Fold: %s' % fold)
            generate_augmented_dataset(task, fold)


def create_selfies_datasets():
    print('\nStarted generating selfies datasets from smiles datasets')
    for folder, files in filter_data_folders(additional_filter='smiles'):
        for smiles_file in filter_data_files(files):
            smiles_full_path = os.path.join(folder, smiles_file)
            write_selfies_file_from_smiles_file(smiles_full_path)


def create_spe_datasets():
    print('\nStarted generating spe datasets from atom-tokenized datasets')
    for folder, files in filter_data_folders(additional_filter='atom'):
        for atom_file in filter_data_files(files):
            atom_full_path = os.path.join(folder, atom_file)
            write_spe_file_from_atom_file(atom_full_path)


def create_w2v_embeddings():
    print('\nStarted generating w2v embeddings for all input combinations')
    for folder, _ in filter_data_folders():
        mol_data = load_rxn_molecules_for_w2v(folder)
        w2v_vectors = train_w2v_vectors(mol_data)
        write_embedding_vectors(folder, w2v_vectors)


def filter_data_folders(additional_filter=''):
   return [(folder, files) for folder, _, files in os.walk(DATA_DIR)
            if 'x1' in folder and 'x10' not in folder
            and '-single' not in folder and '-noreag' not in folder
            and additional_filter in folder]


def filter_data_files(files):
    return [f for f in files
            if any([s in f for s in ['src', 'tgt']])
            and any([s in f for s in ['train', 'test', 'val']])
            and '.txt' in f]


def write_selfies_file_from_smiles_file(smiles_path):
    selfies_path = smiles_path.replace('smiles', 'selfies')
    os.makedirs(os.path.split(selfies_path)[0], exist_ok=True)
    with open(smiles_path, 'r') as f_smiles,\
         open(selfies_path, 'w') as f_selfies:
        progress_bar = tqdm(f_smiles.readlines())
        for smiles in progress_bar:
            progress_bar.set_description('Convert %s to selfies' % smiles_path)
            f_selfies.write(create_selfies_from_smiles(smiles) + '\n')
        

def write_spe_file_from_atom_file(atom_path):
    # Find the correct spe-tokenizer (smiles or selfies)
    if 'smiles' in atom_path:
        spe_path = SPE_ENCODER_PATH_SMILES
    elif 'selfies' in atom_path:
        spe_path = SPE_ENCODER_PATH_SELFIES
    else:
        raise ValueError('There is something wrong with the input data path')
    with open(spe_path, 'r') as spe_file:
        tokenizer = SPE_Tokenizer(codes=spe_file)

    # Build a directory and a path for the new spe-tokenized dataset
    spe_path = atom_path.replace('atom', 'spe')
    os.makedirs(os.path.split(spe_path)[0], exist_ok=True)

    # Build spe-tokenized dataset from the original dataset
    with open(atom_path, 'r') as in_file,\
         open(spe_path, 'w') as out_file:
        progress_bar = tqdm(in_file.readlines())
        for atom_tokenized_rxn in progress_bar:
            progress_bar.set_description('Convert %s to spe' % atom_path)
            rxn = atom_tokenized_rxn.replace(' ', '')
            spe_tokenized_rnx = tokenizer.tokenize(rxn)
            out_file.write(spe_tokenized_rnx + '\n')


def generate_augmented_dataset(task, fold):
    out_fulldir = os.path.join('data', task, 'smiles', 'atom', 'x%s' % fold)
    os.makedirs(out_fulldir, exist_ok=True)
    for split in SPLITS:
        write_smiles_files(out_fulldir, task, fold, split)
        if split == 'test' and task in USPTO_50K_TASKS:
            write_smiles_files(out_fulldir, task, fold, '%s-50k' % split)


def write_smiles_files(out_dir, task, fold, split):
    if split != 'train': fold = 1  # no augmentation for test and valid data
    with open(os.path.join(ORIGINAL_DIR, 'src-%s.txt' % split), 'r') as src_in,\
         open(os.path.join(ORIGINAL_DIR, 'tgt-%s.txt' % split), 'r') as tgt_in,\
         open(os.path.join(out_dir, 'src-%s.txt' % split), 'w') as src_out,\
         open(os.path.join(out_dir, 'tgt-%s.txt' % split), 'w') as tgt_out:

        progress_bar = tqdm(list(zip(src_in.readlines(), tgt_in.readlines())))
        for src, tgt in progress_bar:
            progress_bar.set_description('------ Split %s' % split)
            species = parse_rxn(src, tgt)
            new_src, new_tgt = create_new_sample(task, **species)
            if len(new_src) == 0 or len(new_tgt) == 0: continue
            new_src, new_tgt = augment_sample(new_src, new_tgt, fold)
            src_out.write(new_src + '\n')
            tgt_out.write(new_tgt + '\n')


def parse_rxn(src, tgt):
    src, tgt = src.strip(), tgt.strip()
    if '  >' in src: src = src.replace('  >', ' > ')
    precursors = src.split(' > ')
    reactants, reagents = [p.split(' . ') for p in precursors]
    products = tgt.split(' . ')  # should we pick only the largest?
    return {'reactants': [r for r in reactants if r != ''],\
            'reagents': [r for r in reagents if r != ''],\
            'products': [p for p in products if p != '']}


def create_new_sample(task, reactants, reagents, products):
    if task == 'product-pred':
        new_src = ' . '.join(reactants + reagents)
        new_tgt = ' . '.join(products)
    elif task == 'product-pred-noreag':
        new_src = ' . '.join(reactants)
        new_tgt = ' . '.join(products)
    elif task == 'reactant-pred':
        new_src = ' . '.join(reagents + products)
        new_tgt = ' . '.join(reactants)
    elif task == 'reactant-pred-noreag':
        new_src = ' . '.join(products)
        new_tgt = ' . '.join(reactants)
    elif task == 'reactant-pred-single':
        single_react = random.sample(reactants, 1)  # list
        other_reacts = [r for r in reactants if r != single_react[0]]
        new_src = ' . '.join(other_reacts + reagents + products)
        new_tgt = ' . '.join(single_react)
    elif task == 'reagent-pred':
        new_src = ' . '.join(reactants + products)
        new_tgt = ' . '.join(reagents)
    else:
        raise ValueError('Bad task name')
    return new_src, new_tgt


def augment_sample(src, tgt, fold):
    if fold == 1: return src, tgt  # i.e., no augmentation
    src_smiles_list = src.split(' . ')
    src_augm = [generate_n_equivalent_smiles(s, fold) for s in src_smiles_list]
    src_augm = [list(s) for s in list(zip(*src_augm))]  # re-group molecules
    [random.shuffle(s) for s in src_augm]  # shuffle molecule order
    src_augm = [' . '.join(s) for s in src_augm]  # put back in token string
    tgt_augm = [tgt] * fold  # [' . '.join(sorted(tgt.split(' . ')))] * fold
    return '\n'.join(src_augm), '\n'.join(tgt_augm)


def generate_n_equivalent_smiles(smiles_tokens, n):
    smiles = smiles_tokens.replace(' ', '')
    mol = Chem.MolFromSmiles(smiles)
    new_smiles_list = [smiles]
    trials = 0

    while len(new_smiles_list) < n and trials < 2 * n:
        new_smiles = Chem.MolToSmiles(mol, doRandom=True, canonical=False)
        trials += 1
        if new_smiles not in new_smiles_list:
            new_smiles_list.append(new_smiles)

    if len(new_smiles_list) < n:  # complete list for very short molecules
        factor = n // len(new_smiles_list) + 1
        new_smiles_list = (new_smiles_list * factor)[:n]

    new_smiles_list = [atomwise_tokenizer(s) for s in new_smiles_list]
    return new_smiles_list


def atomwise_tokenizer(smiles):
    # From https://github.com/pschwllr/MolecularTransformer
    # Should also work with selfies (thanks to the [])
    pattern = '(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)'+\
              '|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])'
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smiles)]
    assert smiles == ''.join(tokens)
    return ' '.join(tokens)
    

def canonicalize_smiles(smiles):
    try:
        return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
    except:  # Boost.Python.ArgumentError
        return smiles


def create_selfies_from_smiles(smiles):
    smiles = smiles.replace(' ', '')
    try:
        selfies = sf.encoder(smiles)
    except sf.EncoderError:
        selfies = create_selfies_from_smiles_molecule_by_molecule(smiles)
    return atomwise_tokenizer(selfies)


def create_selfies_from_smiles_molecule_by_molecule(smiles):
    smiles_mols = smiles.split('.')
    selfies_mols = []
    for smiles_mol in smiles_mols:
        try:
            selfies_mol = sf.encoder(smiles_mol)
        except sf.EncoderError:
            selfies_mol = create_seflies_from_canonic_smiles(smiles_mol)
        selfies_mols.append(selfies_mol)
    return '.'.join(selfies_mols)


def create_seflies_from_canonic_smiles(smiles_molecule):
    smiles_molecule = canonicalize_smiles(smiles_molecule)
    try:
        return sf.encoder(smiles_molecule)
    except sf.EncoderError:
        return '?'  # to preserve # molecules / rxn
        # Molecules that didn't work in the USPTO-MIT dataset:
        # O=I(=O)Cl
        # Cl[IH2](Cl)Cl
        # O=[IH2]c1ccccc1
        # F[P-](F)(F)(F)(F)F
        # O=C(O)c1ccccc1I(=O)=O
        # O=C1OI(=O)(O)c2ccccc21
        # S=[Re](=S)(=S)(=S)(=S)(=S)=S
        # CC1(C)O[IH2](C(F)(F)F)c2ccccc21
        # C12C3C4C5C1[Fe]23451678C2C1C6C7C28
        # C12C3C4C5C1[Zr]23451678C2C1C6C7C28
        # O=C(OI(OC(=O)C(F)(F)F)c1ccccc1)C(F)(F)F
        # CC(=O)OI1(OC(C)=O)(OC(C)=O)OC(=O)c2ccccc21
        # Cc1ccc(S(=O)(=O)N=C2CCCC[IH2]2c2ccccc2)cc1
        # O=C(O[IH2](OC(=O)C(F)(F)F)c1ccccc1)C(F)(F)F
        # COc1cc2c(cc1OC)C([PH2](c1ccccc1)(c1ccccc1)c1ccccc1)OC2=O
        # O=c1[nH]c2c3occc3c(F)c(F)c2n1-c1ccc([IH]S(=O)(=O)C2CC2COCc2ccccc2)cc1F


def load_rxn_molecules_for_w2v(data_dir_in):
    # Consider reactions as sets of dot-separated molecules
    # The goal is to share src and tgt embeddings for any task
    with open(os.path.join(data_dir_in, 'src-train.txt'), 'r') as f_src,\
         open(os.path.join(data_dir_in, 'tgt-train.txt'), 'r') as f_tgt:
        sentences = []
        for src, tgt in zip(f_src.readlines(), f_tgt.readlines()):
            molecules = ' . '.join([src.strip(), tgt.strip()])
            sentences.append(molecules.split())
    return sentences


def train_w2v_vectors(sentences):
    model = Word2Vec(vector_size=256, min_count=1, window=5)
    model.build_vocab(corpus_iterable=sentences)  # should check clash with omt
    model.train(corpus_iterable=sentences,
                epochs=model.epochs,
                total_examples=model.corpus_count,
                total_words=model.corpus_total_words)
    return model.wv


def write_embedding_vectors(in_dir, embedding_vectors):
    print('-- Writing embeddings vectors for %s' % in_dir)
    embedding_vectors.save(os.path.join(in_dir, 'w2v.wordvectors'))
    with open(os.path.join(in_dir, 'w2v-embeddings.txt'), 'w') as f:
        for token in embedding_vectors.index_to_key:
            vector = embedding_vectors[token].tolist()
            f.write(token + ' ' + ' '.join(list(map(str, vector))) + '\n')


if __name__ == '__main__':
    main()
    