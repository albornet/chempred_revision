import os


DATA_DIR = os.path.abspath('data')
LOGS_DIR = os.path.abspath('logs')
CONFIGS_DIR = os.path.abspath('configs')
BASE_CONFIG_PATH = os.path.abspath(
    os.path.join('data', 'original', 'base_test.yml'))
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
    config_path, last_ckpt_path, data_path, logs_path, output_path =\
        identify_paths(ckpt_folder, logs_folder, mode)
    to_write = open(BASE_CONFIG_PATH, 'r').read()
    with open(config_path, 'w') as f:
        f.writelines(to_write.replace('$LAST_CKPT_PATH', last_ckpt_path)\
                             .replace('$DATA_PATH', data_path)\
                             .replace('$LOGS_PATH', logs_path)\
                             .replace('$OUTPUT_PATH', output_path))


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
    last_ckpt_path = os.path.join(ckpt_folder, os.listdir(ckpt_folder)[-1])
    if 'roundtrip' in mode:  # exchange input data and prediction model
        data_path = output_path  # path of reactant predictions on test data
        last_ckpt_path = last_ckpt_path.replace('reactant-pred', 'product-pred')
    return config_path, last_ckpt_path, data_path, logs_path, output_path


def reset_test_and_roundtrip_configs(mode):
    for folder, _, files in os.walk(CONFIGS_DIR):
        if '%s.yml' % mode in files:
            to_remove = os.path.join(folder, '%s.yml' % mode)
            os.system('rm %s' % to_remove)


if __name__ == '__main__':
    main()
