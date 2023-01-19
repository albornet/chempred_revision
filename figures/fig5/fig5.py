import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import selfies as sf
import pickle
from itertools import chain
from tqdm.auto import trange
from collections import defaultdict
from typing import List, Tuple, Dict, Union
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


FILE_DIR = os.path.split(__file__)[0]
DATA_DIR = os.path.join(FILE_DIR, '..', '..', 'data')
LOGS_DIR = os.path.join(FILE_DIR, '..', '..', 'logs')
GOLD_PATH = os.path.join(DATA_DIR,
                         'reagent-pred',
                         'smiles',
                         'atom',
                         'x1',
                         'tgt-test.txt')
PRED_PATH = os.path.join(LOGS_DIR,
                         'reagent-pred',
                         'smiles',
                         'atom',
                         'x1',
                         'from-scratch',
                         'test_predictions.txt')
NUM_PRED = 10  # how many predictions per sample
MAX_REAGENTS = 12  # up to how many reagents per reaction the analysis goes
REAGENTS_PER_REACTION = range(1, MAX_REAGENTS + 1)
TOPKS = (1, 3, 5, 10)
LABEL_FONTSIZE = 16
TICK_FONTSIZE = 14
LINE_PARAMS =  {'lw': 1,
                'marker': 'o',
                'markeredgewidth': 1,
                'markeredgecolor': 'k',
                'markersize': 10}


def do_plot():
    # Load predictions and true labels and cluster by number of true reagents
    try:
        data_path = os.path.join(FILE_DIR, '..', 'fig4', 'fig45_data.pickle')
        with open(data_path, 'rb') as f:
            clusters = pickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            'Data for figures 4 and 5 not generated yet.\
             You must set LOAD_DATA to False in figures/fig4/fig4.py')
    
    # Plot figure 5
    fig5_path = os.path.join(FILE_DIR, 'fig5.tiff')
    plot_figure_5(clusters, fig5_path, TOPKS)
    print('- Plotted figure 5 at %s!' % FILE_DIR)
      

def topk_scores(cluster: List[Union[Tuple[List[List[str]]], List[List[str]]]],
                k: int,
                strategy: str = 'set') -> Tuple[float]:
    """
    Compute topk precision, recall and f1-score for a cluster of instances
    
    Parameters:
        cluster: data samples and predictions for one particular # of reagents
        k: number of top predictions to consider as correct
    """
    n_tp, n_predicted, n_true = 0, 0, 0
    for y_true, y_hat in cluster:
        pred_mols = [mol for preds in y_hat[:k] for mol in preds]
        true_mols = [mol for chemeq in y_true for mol in chemeq]
        
        if strategy == 'list':
            n_tp += len([c for c in pred_mols if c in true_mols])
            n_predicted = len(pred_mols)
        elif strategy == 'set':
            n_tp += len(set(pred_mols).intersection(set(true_mols)))
            n_predicted += len(set(pred_mols))
        else:
            raise ValueError('Invalid strategy to compute true positives')
        n_true += len(true_mols)
    
    precision = n_tp / n_predicted
    recall = n_tp / n_true
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    
    return precision, recall, f1


def plot_sample_counts(ax, clusters):
    n_samples = []
    n_unique_reagents = []
    for n in REAGENTS_PER_REACTION:
        cluster = clusters[n]
        unique_true = list(set(chain(*[c[0][0] for c in cluster])))
        n_samples.append(len(cluster))
        n_unique_reagents.append(len(unique_true))
        
    ax_ = ax.twinx()
    l = ax.plot(REAGENTS_PER_REACTION, n_samples, 'C0', **LINE_PARAMS)
    l_ = ax_.plot(REAGENTS_PER_REACTION, n_unique_reagents, 'C1', **LINE_PARAMS)
    ax.set_xticks(REAGENTS_PER_REACTION)
    ax.set_xlabel('Number of true reagents per reaction',
                  fontsize=LABEL_FONTSIZE)
    ax.tick_params(labelsize=TICK_FONTSIZE)
    ax_.tick_params(labelsize=TICK_FONTSIZE)
    ax.set_ylim([0, 8000])
    ax_.set_ylim([0, 800])
    ax.set_ylabel('Number of samples', color='C0', fontsize=LABEL_FONTSIZE)
    ax_.set_ylabel('Number of unique reagents',
                   color='C1',
                   fontsize=LABEL_FONTSIZE)
    ax.grid(which='both')
    ax.legend(l + l_,
              ['Number of samples', 'Number of unique reagents'],
              fontsize=TICK_FONTSIZE)


def plot_rec_prec_f1(ax, clusters, topk):
    scores = [topk_scores(clusters[i], topk) for i in range(1, 13)]
    prec, rec, f1 = [[s[i] for s in scores] for i in range(3)]
    avg_prec, avg_rec, avg_f1 = [sum(v) / len(v) for v in (prec, rec, f1)]
    ax.set_title(
        'Average scores @%s: precision: %.2f, recall: %.2f, f1-score: %.2f\n'
        % (topk, avg_prec, avg_rec, avg_f1),
        fontsize=LABEL_FONTSIZE)
    ax.plot(REAGENTS_PER_REACTION, scores, **LINE_PARAMS)
    ax.set_ylabel('Top-%s accuracy' % topk, fontsize=LABEL_FONTSIZE)
    ax.set_ylim([0, 1])
    y_axis = [y / 10 for y in range(11)]
    ax.set_yticks(y_axis[::2])
    ax.set_yticks(y_axis, minor=True)
    ax.set_xlabel('Data augmentation level', fontsize=LABEL_FONTSIZE)
    ax.set_xticks(REAGENTS_PER_REACTION)
    ax.tick_params(labelsize=TICK_FONTSIZE)
    ax.grid(which='both')
    ax.legend(['precision', 'recall', 'f1-score'], fontsize=TICK_FONTSIZE)


def plot_figure_5(clusters, path, topks, figsize=(9, 9)):
    for topk in topks:
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 1, hspace=0.3)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 0])
        plot_rec_prec_f1(ax1, clusters, topk)
        plot_sample_counts(ax2, clusters)
        plt.savefig(path.replace('.', '-@%s.' % topk), dpi=300)


if __name__ == '__main__':
    do_plot()
