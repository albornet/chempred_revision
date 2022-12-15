import rdkit


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
        preds = all_preds[i * n_preds_per_gold:(i + 1) * n_preds_per_gold]
        if len(preds) > 0:
            print()
            print(gold)
            [print(p) for p in preds]


if __name__ == '__main__':
    main()
    