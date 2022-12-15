import os


DATA_DIR = os.path.join('.', 'data')
LOGS_DIR = os.path.join('.', 'logs')
CONFIG_DIR = os.path.join('.', 'config')
BASE_CONFIG_PATH = os.path.join('data', 'original', 'base_test.yml')
ROUNDTRIP_TASKS = ['reactant-pred', 'reactant-pred-noreag']


def main():
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
    if mode == 'roundtrip'\
        and not any([s in ckpt_folder for s in ROUNDTRIP_TASKS]):        
        return  # round-trip only happens for certain tasks
    config_path, ckpt_path, data_folder =\
        identify_folders_and_paths(ckpt_folder, logs_folder, mode)
    to_write = open(BASE_CONFIG_PATH, 'r').read()
    with open(config_path, 'w') as f:
        f.writelines(to_write.replace('$LAST_CKPT_PATH', ckpt_path)\
                             .replace('$DATA_FOLDER', data_folder)\
                             .replace('$LOGS_FOLDER', logs_folder))


def identify_folders_and_paths(logs_folder, ckpt_folder, mode):
    base_folder = os.path.split(logs_folder)[0]  # remove embed type subfolder
    data_folder = base_folder.replace(LOGS_DIR, DATA_DIR)
    config_folder = base_folder.replace(LOGS_DIR, CONFIG_DIR)
    config_path = os.path.join(config_folder, '%s.yml' % mode)
    ckpt_path = os.path.join(ckpt_folder, os.listdir(ckpt_folder)[-1])  # last
    if mode == 'roundtrip':  # exchange input data and prediction model
        data_folder = logs_folder  # folder for reactant prediction on test data
        ckpt_path = ckpt_path.replace('reactant-pred', 'product-pred')\
                             .replace('reactant-pred-noreag', 'product-pred')
    return config_path, ckpt_path, data_folder


if __name__ == '__main__':
    main()
