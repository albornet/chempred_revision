import os
import re
import csv
from tqdm import tqdm
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


ORIGINAL_DATA_DIR = os.path.abspath('../original')
ORIGINAL_50K_DATA_DIR = os.path.abspath('original')
SPLITS = ['val', 'test', 'train']
STEREO_PATTERNS = {
    '[C@H]': 'C', '[C@@H]': 'C', '[C@]': 'C', '[C@@]': 'C',
    '[S@H]': 'S', '[S@@H]': 'S', '[S@]': 'S', '[S@@]': 'C',
}


def main():
    all_src_lines, all_tgt_lines = [], []
    for split in SPLITS:
        src_lines, tgt_lines = standardize_data(split)
        all_src_lines.extend(src_lines)
        all_tgt_lines.extend(tgt_lines)
    with open(os.path.join(ORIGINAL_DATA_DIR, 'src-test-50k.txt'), 'w') as f:
        f.writelines(all_src_lines)
    with open(os.path.join(ORIGINAL_DATA_DIR, 'tgt-test-50k.txt'), 'w') as f:
        f.writelines(all_tgt_lines)


def standardize_data(split, write_individual_files=False):
    original_data_file = os.path.join(ORIGINAL_50K_DATA_DIR, '%s.csv' % split)
    with open(original_data_file, mode='r') as csv_file:
        src_lines, tgt_lines = [], []
        progress_bar = tqdm(list(csv.DictReader(csv_file, delimiter=',')))
        for row in progress_bar:
            progress_bar.set_description('Standardizing %s file' % split)
            rxn_no_atom_mapping = remove_atom_mapping(row['rxn_smiles'])
            rxn_no_stereo_info = remove_stereo_info(rxn_no_atom_mapping)
            reactants, product = rxn_no_stereo_info.split('>>')
            reactants = atomwise_tokenizer(canonicalize_smiles(reactants))
            product = atomwise_tokenizer(canonicalize_smiles(product))
            src_lines.append(reactants + '  >' + '\n')
            tgt_lines.append(product + '\n')
    if write_individual_files:
        with open('src-%s.txt' % split, 'w') as f: f.writelines(src_lines)
        with open('tgt-%s.txt' % split, 'w') as f: f.writelines(tgt_lines)
    return src_lines, tgt_lines


def remove_atom_mapping(smiles_rxn):
    reactants, product = smiles_rxn.split('>>')
    reactants = Chem.MolFromSmiles(reactants)
    product = Chem.MolFromSmiles(product)
    [a.SetAtomMapNum(0) for a in reactants.GetAtoms()]
    [a.SetAtomMapNum(0) for a in product.GetAtoms()]
    reactants = Chem.MolToSmiles(reactants)
    product = Chem.MolToSmiles(product)
    smiles_rxn = '>>'.join([reactants, product])
    return smiles_rxn


def remove_stereo_info(smiles):
    for pattern, replaced in STEREO_PATTERNS.items():
        smiles = smiles.replace(pattern, replaced)
    return smiles


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


if __name__ == '__main__':
    main()
