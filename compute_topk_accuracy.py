import selfies as sf
from rdkit import Chem
# from rdkit import RDLogger
# RDLogger.DisableLog('rdApp.*')


ALL_KS = [1, 3, 5, 10]


def main():
    pred_path = './example_test_predictions.txt'
    gold_path = './example_tgt-test.txt'
    for k in ALL_KS:
        compute_model_topk_accuracy(pred_path, gold_path, k)


def compute_model_topk_accuracy(pred_path, gold_path, k):
    with open(pred_path, 'r') as p: all_preds = p.readlines()
    with open(gold_path, 'r') as g: all_golds = g.readlines()
    all_preds = [''.join(p.strip().split()) for p in all_preds]
    all_golds = [''.join(g.strip().split()) for g in all_golds]
    n_preds_per_gold = len(all_preds) // len(all_golds)
    for i, gold in enumerate(all_golds):
        preds = all_preds[i * n_preds_per_gold:i * n_preds_per_gold + k]
        if 'selfies' in pred_path:
            preds = [sf.decoder(p) for p in preds]
            gold = sf.decoder(gold)
        preds = [canonicalize_smiles(p) for p in preds]
        gold = canonicalize_smiles(gold)
        for pred in preds:
            pred_molecules = pred.split('.')
            gold_molecules = gold.split('.')
            for gold_molecule in gold_molecules:
                pass


def canonicalize_smiles(smiles):
    try:
        return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
    except:  # Boost.Python.ArgumentError
        return smiles


if __name__ == '__main__':
    main()
