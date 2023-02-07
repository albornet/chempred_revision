import os
import csv
import matplotlib.pyplot as plt
import matplotlib.image as image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


FILE_DIR = os.path.split(__file__)[0]
DATA_DIR = os.path.join(FILE_DIR, '..', '..', 'results')
DATA_AUGMENTATIONS = ('x01', 'x02', 'x05', 'x10', 'x20')
TITLES = {'reac_pred': 'Reactant prediction top-1 accuracy',
          'reac_pred-50k': 'Reactant prediction top-1 accuracy',
          'roundtrip': 'Roundtrip accuracy',
          'roundtrip-50k': 'Roundtrip accuracy'}
X_AXIS = [int(s.split('x')[-1]) for s in DATA_AUGMENTATIONS]
Y_AXIS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
Y_RANGE = (0.2, 1.0)
LABEL_FONTSIZE = 16
TICK_FONTSIZE = 14
N_LEGEND_COLS = 2
SPECS = ('strict_reag+', 'strict_reag-', 'lenient_reag+', 'lenient_reag-')
BASE_DATA = {s: {fold: None for fold in DATA_AUGMENTATIONS} for s in SPECS}
LINE_PARAMS =  {'lw': 1,
                'markeredgewidth': 1,
                'markeredgecolor': 'k',
                'markersize': 10}
BBOXES = {
    'reac_pred': {
        'loc': 'lower center',
        'edgecolor': 'k',
        'framealpha': 1.0,
        'fancybox': False
    },
    'reac_pred-50k': {
        'loc': 'upper center',
        'edgecolor': 'k',
        'framealpha': 1.0,
        'fancybox': False
    },
    'roundtrip': {
        'loc': 'upper center',
        'bbox_to_anchor': [0.535, 0.7],
        'columnspacing': 9.5,
        'frameon': False
    },
    'roundtrip-50k': {
        'loc': 'upper center',
        'bbox_to_anchor': [0.535, 0.7],
        'columnspacing': 9.5,
        'frameon': False
    }
}


def do_plot():
    plot_one_figure()
    plot_one_figure(appendix='-50k')
    print('- Plotted figure 3 at %s!' % FILE_DIR)


def plot_one_figure(appendix=''):
    reactant_file = os.path.join(DATA_DIR, 'results_test%s.csv' % appendix)
    roundtrip_file = os.path.join(DATA_DIR, 'results_roundtrip%s.csv' % appendix)
    data_reactant_pred = get_data(reactant_file)
    data_roundtrip = get_data(roundtrip_file)

    fig = plt.figure(figsize=(9, 9))
    gs = fig.add_gridspec(2, 1, hspace=0.3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])

    plot_one_table(ax1, mode='reac_pred%s' % appendix, data=data_reactant_pred)
    plot_one_table(ax2, mode='roundtrip%s' % appendix, data=data_roundtrip)

    save_path = os.path.join(FILE_DIR, 'fig3%s.png' % appendix)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')


def plot_one_table(ax, mode, data):
    title = TITLES[mode]
    if 'Roundtrip' in title:
        ax.plot(X_AXIS, data['strict_reag+'], 'C1', **LINE_PARAMS,
                marker='o', label='With reagents')
        ax.plot(X_AXIS, data['strict_reag-'], 'C0', **LINE_PARAMS,
                marker='o', label='Without reagents')
    else:
        ax.plot(X_AXIS, data['strict_reag+'], 'C1', **LINE_PARAMS,
                marker='o', label='With reagents (strict)')
        ax.plot(X_AXIS, data['strict_reag-'], 'C0', **LINE_PARAMS,
                marker='o', label='Without reagents (strict)')
        ax.plot(X_AXIS, data['lenient_reag+'], 'C1', **LINE_PARAMS,
                marker='^', label='With reagents (lenient)', linestyle='--')
        ax.plot(X_AXIS, data['lenient_reag-'], 'C0', **LINE_PARAMS,
                marker='^', label='Without reagents (lenient)', linestyle='--')
        
    ax.set_ylabel(title, fontsize=LABEL_FONTSIZE)
    ax.set_ylim(*Y_RANGE)
    ax.set_yticks(Y_AXIS[0::2])
    ax.set_yticks(Y_AXIS, minor=True)
    ax.set_xlabel('Data augmentation level', fontsize=LABEL_FONTSIZE)
    ax.set_xticks(X_AXIS)
    ax.set_xticklabels([s.replace('x0', 'x') for s in DATA_AUGMENTATIONS])
    ax.tick_params(labelsize=TICK_FONTSIZE)
    ax.grid(which='both')
    ax.legend(fontsize=TICK_FONTSIZE, ncol=N_LEGEND_COLS, **BBOXES[mode])
    
    if 'Roundtrip' in title:
        helper = image.imread(os.path.join(FILE_DIR, 'fig3_helper.png'))
        helper_box = OffsetImage(helper, zoom=0.23333)
        helper_ab = AnnotationBbox(helper_box, frameon=True, xy=(10.5, 0.487))
        ax.add_artist(helper_ab)


def get_data(file_path):
    with open(file_path, 'r') as file:
        csv_reader = list(csv.reader(file))[1:]

    data = dict(BASE_DATA)
    for row in csv_reader:
        try:
            task, fmt, tok, emb, fold, acc1, _, _, _, len1, _, _, _ = row
        except ValueError:
            task, fmt, tok, emb, fold, acc1, len1 = row

        if not (fmt == 'smiles' and tok == 'atom' and emb == 'from-scratch'):
            continue

        if task == 'reactant-pred':
            data['strict_reag+'][fold] = float(acc1)
            data['lenient_reag+'][fold] = float(len1)

        if task == 'reactant-pred-noreag':
            data['strict_reag-'][fold] = float(acc1)
            data['lenient_reag-'][fold] = float(len1)

    return {k: list(dict(sorted(v.items())).values()) for k, v in data.items()}


if __name__ == '__main__':
    do_plot()
