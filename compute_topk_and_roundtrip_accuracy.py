import os
import csv 
import selfies as sf
from tqdm import tqdm
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


TOPKS = [1, 3, 5, 10]
HEADERS = ['task', 'format', 'token', 'embed', 'augment'] +\
          ['top-%s' % k for k in TOPKS]
LOGS_DIR = os.path.join('.', 'logs')
DATA_DIR = os.path.join('.', 'data')


def main():
    evaluate_models('test')
    evaluate_models('roundtrip')


def evaluate_models(mode):
    write_result_line(HEADERS, mode, 'w')  # initialize result file
    for folder, _, files in os.walk(LOGS_DIR):
        if '%s_predictions.txt' % mode in files:
            print('Starting %s' % folder)
            pred_path = os.path.join(folder, '%s_predictions.txt' % mode)
            gold_dir = os.path.split(folder)[0].replace(LOGS_DIR, DATA_DIR)
            gold_flag = 'src' if mode == 'roundtrip' else 'tgt'
            gold_path = os.path.join(gold_dir, '%s-test.txt' % gold_flag)
            compute_model_topk_accuracy(pred_path, gold_path, mode)


def compute_model_topk_accuracy(pred_path, gold_path, mode):
    # Retrieve model data and initialize parameters
    all_preds, all_golds = read_pred_and_data(pred_path, gold_path)
    n_preds_per_gold = len(all_preds) // len(all_golds)
    topks = [1] if mode == 'rountrip' else TOPKS
    topk_hits = {k: [] for k in topks}
    
    # Compute all top-k accuracies for this model
    progress_bar = tqdm(list(enumerate(all_golds)))
    for i, gold in progress_bar:
        progress_bar.set_description('Computing %s accuracy' % mode)
        preds = all_preds[i * n_preds_per_gold:(i + 1) * n_preds_per_gold]
        preds, gold = standardize_molecules(preds, gold, pred_path)
        [topk_hits[k].append(compute_topk_hit(preds[:k], gold)) for k in topks]
    
    # Write results for this model in a common file
    _, task, format, token, augment, embed, _ =\
        pred_path.split(LOGS_DIR)[-1].split(os.path.sep)
    model_specs = [task, format, token, embed, augment]
    topk_data = model_specs + [sum(v) / len(v) for v in topk_hits.values()]
    write_result_line(topk_data, mode, 'a')


def read_pred_and_data(pred_path, gold_path):
    with open(pred_path, 'r') as p: all_preds = p.readlines()
    with open(gold_path, 'r') as g: all_golds = g.readlines()
    all_preds = [''.join(p.strip().split()) for p in all_preds]
    all_golds = [''.join(g.strip().split()) for g in all_golds]
    return all_preds, all_golds


def standardize_molecules(preds, gold, pred_path):
    if 'selfies' in pred_path:
        preds = [sf.decoder(p) for p in preds]
        gold = sf.decoder(gold)
    preds = [canonicalize_smiles(p) for p in preds]
    gold = canonicalize_smiles(gold)
    return preds, gold


def compute_topk_hit(preds, gold):
    # Requirement: at least one of preds (list of length k) is accurate
    return any([compute_hit(p.split('.'), gold.split('.')) for p in preds])


def compute_hit(pred_molecules, gold_molecules):
    # Requirement: all gold molecules are in the momdel prediction
    return all([gold in pred_molecules for gold in gold_molecules])


def canonicalize_smiles(smiles):
    try:
        return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
    except:  # Boost.Python.ArgumentError
        return smiles


def write_result_line(content, mode, write_or_append):
    result_flag = 'roundtrip' if mode == 'roundtrip' else 'topk'
    with open('results_%s.csv' % result_flag, write_or_append) as f:
        writer = csv.writer(f); writer.writerow(content)


if __name__ == '__main__':
    main()
