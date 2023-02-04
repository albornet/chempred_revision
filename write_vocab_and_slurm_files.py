import os
import math
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-v', '--vocab', action='store_true')
parser.add_argument('-s', '--slurm', action='store_true')
parser.add_argument('-r', '--reduce', default=1.0, type=float)
args = parser.parse_args()


CONFIGS_DIR = os.path.abspath('configs')
LOGS_DIR = os.path.abspath('logs')
DATA_DIR = os.path.abspath('data')
SLURM_DIR = os.path.abspath('slurm')
BASE_SLURM_PATH = os.path.join(DATA_DIR, 'original', 'base_slurm.sh')
VOCAB_SCRIPT = 'python open-nmt/build_vocab.py'
DO_VOCAB = args.vocab
DO_SLURM = args.slurm
DATA_REDUCTION_FACTOR = args.reduce  # 0.0625, 0.125, 0.25, 0.5, 1.0 (no reduce)
FOLDS = [1, 2, 5, 10, 20]
TEST_MODES = ['test', 'test-50k', 'roundtrip', 'roundtrip-50k']
ROUNDTRIP_SPECS = ['atom', 'smiles', 'from-scratch']
ROUNDTRIP_TASKS = ['reactant-pred', 'reactant-pred-noreag']
USPTO_50K_TASKS = [
    'product-pred',
    'product-pred-noreag',
    'reactant-pred',
    'reactant-pred-noreag',
    'reactant-pred-single'
]


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
            if DO_VOCAB:
                os.system('%s -config %s -n_sample -1' % (VOCAB_SCRIPT,
                                                          config_path))
            if DO_SLURM:
                slurm_path = write_train_slurm_script(config_path, fold)
                write_test_slurm_file(slurm_path)


def write_train_slurm_script(config_path, fold):
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
    slurm_subdir = os.path.join(SLURM_DIR, 'train', fold_subdir)
    slurm_path = '-'.join(config_path.split(CONFIGS_DIR)[1]
                                     .split(os.path.sep)[1:])
    
    if fold == 1 and 'smiles-atom-x1-from-scratch' not in slurm_path:
        slurm_subdir = slurm_subdir.replace(fold_subdir, 'input_scheme_exp')
    else:
        slurm_subdir = slurm_subdir.replace(fold_subdir,
                                            'data_augment_exp_%s' % fold_subdir)
    
    os.makedirs(slurm_subdir, exist_ok=True)
    slurm_full_path = os.path.join(slurm_subdir,
                                   slurm_path.replace('.yml', '.sh'))
    
    with open(slurm_full_path, 'w') as f:
        f.writelines(slurm_script)
    return slurm_full_path


def compute_runtime(fold):
    n_hours_base = 24  # approximate runtime for 1 fold data augmentation
    n_hours_max = 7 * 24  # 7 days = max runtime on baobab
    n_hours = n_hours_base +\
              (n_hours_max - n_hours_base) *\
              math.log(((math.e - 1) * fold + (20 - math.e)) / 19)
    n_hours *= DATA_REDUCTION_FACTOR  # for subsampbled training datasets
    return '%s-%s:00:00' % divmod(int(n_hours), 24)


def write_test_slurm_file(train_path):
    train_dir, train_file = os.path.split(train_path)
    for mode in TEST_MODES:
        if 'roundtrip' in mode\
        and (not any([s in train_file for s in ROUNDTRIP_TASKS])\
             or '-single' in train_file):
            continue
        
        if 'roundtrip' in mode\
        and not all([s in train_file for s in ROUNDTRIP_SPECS]):
            continue

        if '50k' in mode\
        and not any([s in train_file for s in USPTO_50K_TASKS]):
            continue
        
        with open(train_path, 'r') as f: lines = f.readlines()
        time_hours = '1:30' if 'roundtrip' in mode else '0:30'
        time_line = '#SBATCH --time=0-%s:00\n' % time_hours
        lines = [time_line if '--time' in l else l for l in lines]
        to_write = ''.join(lines).replace('train.py', 'translate.py')\
                                 .replace('train.yml', '%s.yml' % mode)\
                                 .replace('slurm.log', 'slurm-%s.log' % mode)\
                                 .replace('slurm.err', 'slurm-%s.err' % mode)

        
        test_dir = train_dir.replace('train%s' % os.path.sep,
                                     '%s%s' % (mode.strip('-50k'), os.path.sep))
        test_file = train_file.replace('train.sh', '%s.sh' % mode)
        test_path = os.path.join(test_dir, test_file)
        os.makedirs(test_dir, exist_ok=True)
        with open(test_path, 'w') as f: f.write(to_write)
        

if __name__ == '__main__':
    main()
