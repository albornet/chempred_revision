import os


DATA_DIR = os.path.join('.', 'data')
LOGS_DIR = os.path.join('.', 'logs')
CONFIG_DIR = os.path.join('.', 'config')
BASE_CONFIG_PATH = os.path.join('data', 'original', 'base_$MODE.yml')
ROUNDTRIP_TASKS = ['reactant-pred', 'reactant-pred-noreag']
ROUNDTRIP_SPECS = ['atom', 'smiles', 'from-scratch']


def main():
    reset_test_and_roundtrip_configs()
    generate_config_folder(mode='test')
    print('Testing configuration files generated!')
    generate_config_folder(mode='roundtrip')
    print('Roundtrip configuration files generated')


def generate_config_folder(mode):
    for folder, _, files in os.walk(LOGS_DIR):
        if os.path.split(folder)[-1] == 'ckpts' and len(files) > 0:
            ckpt_folder = folder
            logs_folder = os.path.split(ckpt_folder)[0]
            write_config_file(ckpt_folder, logs_folder, mode)


def write_config_file(ckpt_folder, logs_folder, mode):
    if mode == 'roundtrip':
        if not check_if_roundtrip_should_be_run(ckpt_folder): return
    config_path, ckpt_path, data_folder =\
        identify_folders_and_paths(ckpt_folder, logs_folder, mode)
    to_write = open(BASE_CONFIG_PATH.replace('$MODE', mode), 'r').read()
    with open(config_path, 'w') as f:
        f.writelines(to_write.replace('$LAST_CKPT_PATH', ckpt_path)\
                             .replace('$DATA_FOLDER', data_folder)\
                             .replace('$LOGS_FOLDER', logs_folder))


def check_if_roundtrip_should_be_run(folder):
    if not any([s + os.path.sep in folder for s in ROUNDTRIP_TASKS]):
        return False
    if not all([s in folder for s in ROUNDTRIP_SPECS]):
        return False
    return True


def identify_folders_and_paths(ckpt_folder, logs_folder, mode):
    data_folder = os.path.split(logs_folder)[0]  # remove embed type subfolder
    data_folder = data_folder.replace(LOGS_DIR, DATA_DIR)
    config_folder = logs_folder.replace(LOGS_DIR, CONFIG_DIR)
    config_path = os.path.join(config_folder, '%s.yml' % mode)
    ckpt_path = os.path.join(ckpt_folder, os.listdir(ckpt_folder)[-1])  # last
    if mode == 'roundtrip':  # exchange input data and prediction model
        data_folder = logs_folder  # folder for reactant prediction on test data
        ckpt_path = ckpt_path.replace('reactant-pred', 'product-pred')  #  .replace('-noreag', '')
    return config_path, ckpt_path, data_folder


def reset_test_and_roundtrip_configs():
    for folder, _, files in os.walk(CONFIG_DIR):
        if 'roundtrip.yml' in files:
            to_remove = os.path.join(folder, 'roundtrip.yml')
            os.system('rm %s' % to_remove)
        if 'test.yml' in files:
            to_remove = os.path.join(folder, 'test.yml')
            os.system('rm %s' % to_remove)


if __name__ == '__main__':
    main()
