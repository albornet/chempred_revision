import os
import csv
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


FILE_DIR = os.path.split(__file__)[0]
DATA_DIR = os.path.join(FILE_DIR, '..', '..', 'results')
DATA_AUGMENTATIONS = ('x01', 'x02', 'x05', 'x10', 'x20')
TOPKS = [1, 3, 5, 10]
X_AXIS = [int(s.split('x')[-1]) for s in DATA_AUGMENTATIONS]
Y_AXIS = [y / 10 for y in range(10 + 1)]
Y_RANGE = (0.2, 1.0)
LABEL_FONTSIZE = 16
TICK_FONTSIZE = 14
BBOXES = {
    'test': {
        'bbox_to_anchor': [0.97, 0.47],
        'loc': 'upper right',
        'borderaxespad': 0
    },
    'test-50k': {
        'bbox_to_anchor': [0.97, 0.05],
        'loc': 'lower right',
        'borderaxespad': 0
    },
}
TASKS = [
    'product-pred',
    'reactant-pred',
    'reactant-pred-single',
    'product-pred-noreag',
    'reactant-pred-noreag',
    'reagent-pred'
]
COLORS = {
    'product-pred': 'C1',
    'product-pred-noreag': 'C1',
    'reactant-pred': 'C2',
    'reactant-pred-noreag': 'C2',
    'reactant-pred-single': 'C2',
    'reagent-pred': 'C3'
}
MARKERS = {
    'product-pred': 'o',
    'product-pred-noreag': '^',
    'reactant-pred': 'o',
    'reactant-pred-noreag': '^',
    'reactant-pred-single': '>',
    'reagent-pred': 'o'    
}
LINE_PARAMS =  {
    'lw': 1,
    'markeredgewidth': 1,
    'markeredgecolor': 'k',
    'markersize': 8
}


def do_plot():
    for topk in TOPKS:
        plot_one_figure(topk=topk)
        plot_one_figure(topk=topk, appendix='-50k')


def plot_one_figure(topk, appendix=''):
    result_path = os.path.join(DATA_DIR, 'results_test%s.csv' % appendix)
    data = get_data(result_path)  # dictionary, keys are tasks
    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot()
    plot_one_table(ax, mode='test%s' % appendix, data=data, topk=topk)
    save_path = os.path.join(FILE_DIR, 'fig2%s-top%s.tiff' % (appendix, topk))
    plt.savefig(save_path, dpi=300, bbox_inches='tight')


def plot_one_table(ax, mode, data, topk):
    for task in data.keys():
        to_plot = [d[TOPKS.index(topk)] for d in data[task]]
        color, marker = COLORS[task], MARKERS[task]
        ax.plot(X_AXIS, to_plot, color, **LINE_PARAMS, marker=marker, label=task)

    ax.set_ylabel('Top-%s accuracy' % topk, fontsize=LABEL_FONTSIZE)
    ax.set_ylim(*Y_RANGE)
    ax.set_yticks(Y_AXIS[0::2])
    ax.set_yticks(Y_AXIS, minor=True)
    ax.set_xlabel('Data augmentation level', fontsize=LABEL_FONTSIZE)
    ax.set_xticks(X_AXIS)
    ax.set_xticklabels([s.replace('x0', 'x') for s in DATA_AUGMENTATIONS])
    ax.tick_params(labelsize=TICK_FONTSIZE)
    ax.grid(which='both')
    bbox_params = BBOXES[mode]
    bbox_params['bbox_to_anchor'][1] += (topk - 1) / topk ** 2.5
    ax.legend(fontsize=TICK_FONTSIZE, ncol=2, **bbox_params)


def get_data(file_path):
    with open(file_path, 'r') as file:
        csv_reader = list(csv.reader(file))[1:]
    
    tasks = list(TASKS)
    if '-50k' in os.path.split(file_path)[-1]: tasks.remove('reagent-pred')
    data = {k: {} for k in tasks}
    for row in csv_reader:
        task, fmt, tok, emb, fold, acc1, acc3, acc5, acc10, _, _, _, _ = row
        if not (fmt == 'smiles' and tok == 'atom' and emb == 'from-scratch'):
            continue
        if task in data.keys():
            data[task][fold] = [float(a) for a in [acc1, acc3, acc5, acc10]]
            
    return {k: list(dict(sorted(v.items())).values()) for k, v in data.items()}


if __name__ == '__main__':
    do_plot()
