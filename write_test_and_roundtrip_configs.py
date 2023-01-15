import os


DATA_DIR = os.path.abspath('data')
LOGS_DIR = os.path.abspath('logs')
CONFIGS_DIR = os.path.abspath('configs')
BASE_CONFIG_PATH = os.path.abspath(
    os.path.join('data', 'original', 'base_test.yml'))
CKPT_SELECTION_MODE = 'last'  # 'first', 'last', 'best'
MODES = ['test', 'test-50k', 'roundtrip', 'roundtrip-50k']
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
    for mode in MODES:
        reset_test_and_roundtrip_configs(mode=mode)
        generate_config_folder(mode=mode)
        print('Configuration files generated for %s!' % mode)
        

def generate_config_folder(mode):
    for folder, _, files in os.walk(LOGS_DIR):
        if os.path.split(folder)[-1] == 'ckpts' and len(files) > 0:
            ckpt_folder = folder
            logs_folder = os.path.split(ckpt_folder)[0]  # 1 arborescence higher
            write_config_file(ckpt_folder, logs_folder, mode)


def write_config_file(ckpt_folder, logs_folder, mode):
    if 'roundtrip' in mode:
        if not check_if_roundtrip_should_be_run(ckpt_folder): return
    if '50k' in mode:
        if not check_if_50k_should_be_run(ckpt_folder): return
    config_path, ckpt_path, data_path, logs_path, output_path =\
        identify_paths(ckpt_folder, logs_folder, mode)
    n_best = '1' if 'roundtrip' in mode else '10'
    to_write = open(BASE_CONFIG_PATH, 'r').read()
    with open(config_path, 'w') as f:
        f.writelines(to_write.replace('$CKPT_PATH', ckpt_path)\
                             .replace('$DATA_PATH', data_path)\
                             .replace('$LOGS_PATH', logs_path)\
                             .replace('$OUTPUT_PATH', output_path)\
                             .replace('$N_BEST', n_best))


def check_if_roundtrip_should_be_run(folder):
    if not any([s + os.path.sep in folder for s in ROUNDTRIP_TASKS]):
        return False
    if not all([s in folder for s in ROUNDTRIP_SPECS]):
        return False
    return True


def check_if_50k_should_be_run(folder):
    if not any([s + os.path.sep in folder for s in USPTO_50K_TASKS]):
        return False
    return True


def identify_paths(ckpt_folder, logs_folder, mode):
    data_folder = os.path.split(logs_folder)[0]  # remove embed type subfolder
    data_folder = data_folder.replace(LOGS_DIR, DATA_DIR)
    data_path = os.path.join(data_folder, 'src-%s.txt' % mode)
    config_folder = logs_folder.replace(LOGS_DIR, CONFIGS_DIR)
    config_path = os.path.join(config_folder, '%s.yml' % mode)
    logs_path = os.path.join(logs_folder, '%s.log' % mode)
    output_path = os.path.join(logs_folder, '%s_predictions.txt' % mode)
    if 'roundtrip' in mode:  # exchange input data and prediction model
        write_roundtrip_data(data_path, output_path)
        ckpt_folder = ckpt_folder.replace('reactant-pred', 'product-pred')
        # ckpt_folder = ckpt_folder.replace('reactant-pred-noreag',
        #                                   'product-pred-noreag')\
        #                          .replace('reactant-pred',
        #                                   'product-pred-noreag')
    ckpt_path = select_ckpt_path(ckpt_folder)
    return config_path, ckpt_path, data_path, logs_path, output_path


def select_ckpt_path(ckpt_folder):
    if CKPT_SELECTION_MODE == 'first':
        return os.path.join(ckpt_folder, os.listdir(ckpt_folder)[0])
    elif CKPT_SELECTION_MODE == 'last':
        return os.path.join(ckpt_folder, os.listdir(ckpt_folder)[-1])
    elif CKPT_SELECTION_MODE == 'best':
        logs_folder = os.path.split(ckpt_folder)[0]
        with open(os.path.join(logs_folder, 'train_logs.log'), 'r') as f:
            best_ckpt_line = f.readlines()[-3]
            if 'Best model found at step' in best_ckpt_line:
                step = best_ckpt_line.split('step ')[-1].strip()
                return os.path.join(ckpt_folder, 'model_step_%s.pt' % step)
            else:  # best ckpt may be the last deleted one (so, closest = first)
                return os.path.join(ckpt_folder, os.listdir(ckpt_folder)[0])
    else:
        raise ValueError('Invalid mode for checkpoint selection')


def write_roundtrip_data(data_path, output_path):
    reac_pred_path = output_path.replace('roundtrip', 'test')
    with open(reac_pred_path, 'r') as f: reac_pred_lines = f.readlines()
    if '-noreag' not in output_path and '-50k' not in output_path:
        reac_reag_path = os.path.join(DATA_DIR, 'original', 'src-test.txt')
        with open(reac_reag_path, 'r') as f_r,\
             open(data_path, 'w') as f_w:
            for i, reac_reag_line in enumerate(f_r.readlines()):
                reag_line = reac_reag_line.split(' >')[1].strip()
                reac_pred_matches = reac_pred_lines[i * 10:(i + 1) * 10]
                if len(reag_line) > 0:
                    reac_pred_matches = [' . '.join([p.strip(), reag_line])
                                         + '\n' for p in reac_pred_matches]
                f_w.writelines(reac_pred_matches)
    else:
        with open(data_path, 'w') as f_w:
            f_w.writelines(reac_pred_lines)


def reset_test_and_roundtrip_configs(mode):
    for folder, _, files in os.walk(CONFIGS_DIR):
        if '%s.yml' % mode in files:
            to_remove = os.path.join(folder, '%s.yml' % mode)
            os.remove(to_remove)


if __name__ == '__main__':
    main()
