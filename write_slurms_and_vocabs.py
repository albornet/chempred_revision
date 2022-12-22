import os


CONFIG_DIR = os.path.join('.', 'config')
LOGS_DIR = os.path.join('.', 'logs')
SLURM_DIR = os.path.join('.', 'slurm')
VOCAB_SCRIPT = 'python open-nmt/build_vocab.py'
WRITE_VOCAB = False
BASE_SLURM_PATH = os.path.join('.', 'data', 'original', 'base_slurm.sh')
FOLDS = [1, 2, 5, 10, 20]


def main():
    # Separate folders by data augmentation fold (take different times)
    for fold in FOLDS:
        write_slurms_for_one_data_augmentation_level(fold)
    print('Slurm bash scripts generated!')


def write_slurms_for_one_data_augmentation_level(fold):
    for folder, _, files in os.walk('./config'):
        if 'train.yml' in files\
        and str(fold) in folder and str(10 * fold) not in folder:
            config_path = os.path.join(folder, 'train.yml')
            if WRITE_VOCAB:
                os.system(
                    '%s -config %s -n_sample -1' % (VOCAB_SCRIPT, config_path))
            write_slurm_script(config_path, fold)


def write_slurm_script(config_path, fold):
    with open(BASE_SLURM_PATH) as f: slurm_script = f.read()
    logs_subdir = os.path.join(
        LOGS_DIR,
        *config_path.split(CONFIG_DIR)[1].split(os.path.sep)[:-1])
    slurm_logs_path = os.path.join(logs_subdir, 'slurm.log')
    slurm_errs_path = os.path.join(logs_subdir, 'slurm.err')
    slurm_script = slurm_script.replace('$CONFIG_PATH', config_path)\
                               .replace('$SLURM_LOGS_PATH', slurm_logs_path)\
                               .replace('$SLURM_ERRS_PATH', slurm_errs_path)
    fold_subdir = 'x%s' % fold
    slurm_subdir = os.path.join(SLURM_DIR, fold_subdir)
    slurm_path = '-'.join(config_path.split(CONFIG_DIR)[1]
                                     .split(os.path.sep)[1:])
    if fold == 1 and 'smiles-atom-x1-from-scratch' not in slurm_path:
        slurm_subdir = slurm_subdir.replace(fold_subdir, 'input_scheme_exp')
    else:
        slurm_subdir = slurm_subdir.replace(fold_subdir,
                                            'data_augment_exp_%s' % fold_subdir)
    os.makedirs(slurm_subdir, exist_ok=True)
    slurm_path = os.path.join(slurm_subdir, slurm_path.replace('.yml', '.sh'))
    with open(slurm_path, 'w') as f: f.writelines(slurm_script)


if __name__ == '__main__':
    main()
