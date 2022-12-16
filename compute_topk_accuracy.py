import os
import selfies as sf
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


ALL_KS = [1, 3, 5, 10]
LOGS_DIR = os.path.join('.', 'config')
DATA_DIR = os.path.join('.', 'data')


def main(debug_mode=False):
    if debug_mode:
        debug(); return
    for folder, _, files in os.walk(LOGS_DIR):
        if '%s.yml' % mode in files:
            config_path = os.path.join(folder, '%s.yml' % mode)
    

def debug():
    pred_path = './example_test_predictions.txt'
    gold_path = './example_tgt-test.txt'
    print('Model: %s' % pred_path)
    for k in ALL_KS:
        compute_model_topk_accuracy(pred_path, gold_path, k)


def compute_model_topk_accuracy(pred_path, gold_path, k):
    all_preds, all_golds = read_pred_and_data(pred_path, gold_path)
    n_preds_per_gold = len(all_preds) // len(all_golds)
    topk_hits = []
    for i, gold in enumerate(all_golds):
        preds = all_preds[i * n_preds_per_gold:i * n_preds_per_gold + k]
        preds, gold = standardize_molecules(preds, gold, pred_path)
        topk_hits.append(compute_topk_hit(preds, gold))
    topk_accuracy = sum(topk_hits) / len(topk_hits)
    print('-- Top-%s accuracy: %s' % (k, topk_accuracy))


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
    return any([compute_hit(pred.split('.'), gold.split('.')) for pred in preds])


def compute_hit(pred_molecules, gold_molecules):
    # Requirement: all gold molecules are in the momdel prediction
    return all([gold in pred_molecules for gold in gold_molecules])


def canonicalize_smiles(smiles):
    try:
        return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
    except:  # Boost.Python.ArgumentError
        return smiles


if __name__ == '__main__':
    main()
