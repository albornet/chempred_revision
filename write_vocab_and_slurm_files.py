import os
import math
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-v', '--vocab', action='store_true')
parser.add_argument('-s', '--slurm', action='store_true')
args = parser.parse_args()


CONFIGS_DIR = os.path.abspath('configs')
LOGS_DIR = os.path.abspath('logs')
DATA_DIR = os.path.abspath('data')
SLURM_DIR = os.path.abspath('slurm')
BASE_SLURM_PATH = os.path.join(DATA_DIR, 'original', 'base_slurm.sh')
VOCAB_SCRIPT = 'python open-nmt/build_vocab.py'
DO_VOCAB = args.vocab
DO_SLURM = args.slurm
FOLDS = [1, 2, 5, 10, 20]


def main():
    # Separate folders by data augmentation fold (take different times)
    for fold in FOLDS:
        write_vocabs_and_slurms_for_one_data_augmentation_level(fold)
    if DO_VOCAB: print('Vocab files generated!')
    if DO_SLURM: print('Slurm bash scripts generated!')
    if not DO_VOCAB and not DO_SLURM:
        raise ValueError('Use this script with one or both of the following '\
            'arguments:\n-v to build vocabularies\n-s to write slurm scripts')


def write_vocabs_and_slurms_for_one_data_augmentation_level(fold):
    for folder, _, files in os.walk(CONFIGS_DIR):
        if 'train.yml' in files\
        and str(fold) in folder and str(10 * fold) not in folder:
            config_path = os.path.join(folder, 'train.yml')
            if DO_VOCAB: os.system('%s -config %s -n_sample -1' %\
                                      (VOCAB_SCRIPT, config_path))
            if DO_SLURM: write_slurm_script(config_path, fold)


def write_slurm_script(config_path, fold):
    with open(BASE_SLURM_PATH) as f: slurm_script = f.read()
    logs_subdir = os.path.join(LOGS_DIR, *config_path.split(CONFIGS_DIR)[1]
                                                     .split(os.path.sep)[:-1])
    slurm_log_path = os.path.join(logs_subdir, 'slurm.log')
    slurm_err_path = os.path.join(logs_subdir, 'slurm.err')
    slurm_runtime = compute_runtime(fold)
    slurm_script = slurm_script.replace('$CONFIG_PATH', config_path)\
                               .replace('$SLURM_LOG_PATH', slurm_log_path)\
                               .replace('$SLURM_ERR_PATH', slurm_err_path)\
                               .replace('$SLURM_RUNTIME', slurm_runtime)
    fold_subdir = 'x%s' % fold
    slurm_subdir = os.path.join(SLURM_DIR, fold_subdir)
    slurm_path = '-'.join(config_path.split(CONFIGS_DIR)[1]
                                     .split(os.path.sep)[1:])
    if fold == 1 and 'smiles-atom-x1-from-scratch' not in slurm_path:
        slurm_subdir = slurm_subdir.replace(fold_subdir, 'input_scheme_exp')
    else:
        slurm_subdir = slurm_subdir.replace(fold_subdir,
                                            'data_augment_exp_%s' % fold_subdir)
    os.makedirs(slurm_subdir, exist_ok=True)
    slurm_path = os.path.join(slurm_subdir, slurm_path.replace('.yml', '.sh'))
    with open(slurm_path, 'w') as f: f.writelines(slurm_script)


def compute_runtime(fold):
    n_hours_base = 48  # approximate runtime for 1 fold data augmentation
    n_hours_max = 7 * 24  # 7 days = max runtime on baobab
    n_hours = n_hours_base +\
              (n_hours_max - n_hours_base) *\
              math.log(((math.e - 1) * fold + (20 - math.e)) / 19)
    return '%s-%s:00:00' % divmod(int(n_hours), 24)


if __name__ == '__main__':
    main()
