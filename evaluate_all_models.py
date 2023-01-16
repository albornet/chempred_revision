import os
import csv
import selfies as sf
from multiprocessing import Pool
from tqdm import tqdm
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


LOGS_DIR = os.path.abspath('logs')
DATA_DIR = os.path.abspath('data')
RESULTS_DIR = os.path.abspath('results')
KS = [1, 3, 5, 10]
MODES = ['test', 'test-50k', 'roundtrip', 'roundtrip-50k']
SPECS = ['task', 'format', 'token', 'embed', 'augment']
HEADERS = SPECS + ['top-%s' % k for k in KS] + ['lenient-%s' % k for k in KS]


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    for mode in MODES:
        evaluate_models(mode)


def evaluate_models(mode):
    result_file_path = os.path.join(RESULTS_DIR, 'results_%s.csv' % mode)
    write_result_line(result_file_path, HEADERS, 'w')  # initialize result file
    args_to_run = []
    for folder, _, files in os.walk(LOGS_DIR):
        if '%s_predictions.txt' % mode in files:
            args_to_run.append((folder, result_file_path, mode))
    n_cpus_used = max(1, os.cpu_count() // 4)
    with Pool(n_cpus_used) as pool:
        pool.map(evaluate_one_model, args_to_run)
    sort_csv(result_file_path, sort_column_order=[0, 1, 2, 4, 3])


def evaluate_one_model(args):
    folder, write_path, mode = args
    print('Starting %s for %s' % (folder, mode))
    pred_path = os.path.join(folder, '%s_predictions.txt' % mode)
    gold_dir = os.path.split(folder)[0].replace(LOGS_DIR, DATA_DIR)
    if not 'noreag' in folder and 'roundtrip' in mode:  # only predict product
        gold_dir = gold_dir.replace('reactant-pred', 'reactant-pred-noreag')
    gold_flag = 'src' if 'roundtrip' in mode else 'tgt'
    mode_flag = '-50k' if '50k' in mode else ''
    gold_path = os.path.join(gold_dir, '%s-test%s.txt' % (gold_flag, mode_flag))
    compute_model_topk_accuracy(write_path, pred_path, gold_path, mode)


def compute_model_topk_accuracy(write_path, pred_path, gold_path, mode):
    # Retrieve model data and initialize parameters
    all_preds, all_golds = read_pred_and_data(pred_path, gold_path)
    n_preds_per_gold = len(all_preds) // len(all_golds)
    topk_hits = {k: [] for k in KS}  # for strict accuracy
    lenk_hits = {k: [] for k in KS}  # for lenient accuracy
    
    # Compute all top-k accuracies for this model
    progress_bar = tqdm(list(enumerate(all_golds)))
    for i, gold in progress_bar:
        progress_bar.set_description('Computing %s accuracy' % mode)
        preds = all_preds[i * n_preds_per_gold:(i + 1) * n_preds_per_gold]
        preds, gold = standardize_molecules(preds, gold, pred_path)
        [topk_hits[k].append(
            compute_topk_hit(preds[:k], gold, mode='all')) for k in KS]
        [lenk_hits[k].append(
            compute_topk_hit(preds[:k], gold, mode='any')) for k in KS]
    
    # Write results for this model in a common file
    _, task, format, token, augment, embed, _ =\
        pred_path.split(LOGS_DIR)[-1].split(os.path.sep)
    augment = 'x%02i' % int(augment.split('x')[-1])  # format for sorting
    model_specs = [task, format, token, embed, augment]
    topk_data = [sum(v) / len(v) for v in topk_hits.values()]
    lenk_data = [sum(v) / len(v) for v in lenk_hits.values()]
    result_line = model_specs + topk_data + lenk_data
    write_result_line(write_path, result_line, 'a')


def write_result_line(file_path, content, write_or_append):
    with open(file_path, write_or_append) as f:
        writer = csv.writer(f); writer.writerow(content)


def read_pred_and_data(pred_path, gold_path):
    with open(pred_path, 'r') as p: all_preds = p.readlines()
    with open(gold_path, 'r') as g: all_golds = g.readlines()
    all_preds = [''.join(p.strip().split()) for p in all_preds]
    all_golds = [''.join(g.strip().split()) for g in all_golds]
    return all_preds, all_golds


def compute_topk_hit(preds, gold, mode='strict'):
    """ Compute whether a list of top-k best predictions contains a correct one
    Args:
        - preds: list of k best predictions, strings of '.'- separated molecules
        - gold: ground truth basis for hit computation (same type of string)
        - mode: whether all or any molecule(s) should be correctly predicted
    """
    if mode == 'all':
        # Requirement: all gold molecules are in the model prediction
        hit_fn = lambda gold, pred: all([g in pred for g in gold])
    elif mode == 'any':
        # Requirement: any of the gold molecules are in the model prediction
        hit_fn = lambda gold, pred: any([g in pred for g in gold])
    elif mode == 'strict':
         # Requirement: exact match between gold and prediction molecules
         hit_fn = lambda gold, pred: sorted(pred) == sorted(gold)
    else:
        raise ValueError('Incorrect mode for compute hit function')
    # Requirement: at least one of preds (list of length k) deserves a hit
    return any([hit_fn(pred.split('.'), gold.split('.')) for pred in preds])


def standardize_molecules(preds, gold, pred_path):
    if 'selfies' in pred_path:
        # preds = [sf.decoder(p) for p in preds]
        # gold = sf.decoder(gold)
        preds = [create_smiles_from_selfies(p) for p in preds]
        gold = create_smiles_from_selfies(gold)
    preds = [canonicalize_smiles(p) for p in preds]
    gold = canonicalize_smiles(gold)
    return preds, gold


def canonicalize_smiles(smiles):
    try:
        return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
    except:  # Boost.Python.ArgumentError
        return smiles


def create_smiles_from_selfies(selfies):
    try:
        return sf.decoder(selfies)
    except sf.DecoderError:
        selfies_mols = selfies.split('.')
        smiles_mols = []
        for selfies_mol in selfies_mols:
            try:
                smiles_mol = sf.decoder(selfies_mol)
            except sf.DecoderError:
                smiles_mol = '?'
            smiles_mols.append(smiles_mol)
        return '.'.join(smiles_mols)


def sort_csv(filepath, sort_column_order):
    with open(filepath, "r") as f:
        reader = csv.reader(f)
        headers = next(reader)
        rows = list(reader)

    rows.sort(key=lambda row: [row[column] for column in sort_column_order])   
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


if __name__ == '__main__':
    main()
