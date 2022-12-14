import os


CONFIG_DIR = os.path.join('./config')
LOGS_DIR = os.path.join('./logs')
N_BASE_STEPS = 500000
EMBED_TYPES = ['from-scratch', 'pre-trained']
W2V_TEXT = """
# Add pre-trained embeddings
both_embeddings: ./data/$SPEC_FOLDER/w2v-embeddings.txt
embeddings_type: word2vec
word_vec_size: 256
freeze_word_vecs_enc: True
freeze_word_vecs_dec: True
"""


def main():
    # generate_data_based_config_files()
    generate_round_trip_based_config_files()
    print('Done!')


def generate_data_based_config_files():
    for folder, subfolders, _ in os.walk('data'):
        if len(subfolders) == 0:
            if 'x1' in folder and 'x10' not in folder\
            and '-single' not in folder and '-noreag' not in folder:
                setup_train_and_test_configs(folder, EMBED_TYPES)
            elif 'original' not in folder:
                setup_train_and_test_configs(folder, [EMBED_TYPES[0]])


def generate_round_trip_based_config_files():
    reactant_pred_folder = os.path.join('data', 'reactant-pred')
    for folder, subfolders, _ in os.walk(reactant_pred_folder):
        if len(subfolders) == 0:
            spec_path = os.path.join(*folder.split(os.path.sep)[1:])
            config_folder = create_folder(CONFIG_DIR, spec_path, 'from-scratch')
            write_config_file(config_folder, 'test')


def setup_train_and_test_configs(data_folder, embed_types_for_this_data):
    spec_path = os.path.join(*data_folder.split(os.path.sep)[1:])
    for embed_type in embed_types_for_this_data:
        config_folder = create_folder(CONFIG_DIR, spec_path, embed_type)
        logs_folder = create_folder(LOGS_DIR, spec_path, embed_type)
        os.makedirs(config_folder, exist_ok=True)
        os.makedirs(logs_folder, exist_ok=True)
        write_config_file(config_folder, 'train')
        write_config_file(config_folder, 'test')


def create_folder(base_dir, spec_path, embed_type):
    folder = os.path.join(base_dir, spec_path, embed_type)
    os.makedirs(folder, exist_ok=True)
    return folder


def create_config_folder(spec_folder, embed_type):
    config_folder = os.path.join(CONFIG_DIR, spec_folder, embed_type)
    logs_folder = os.path.join(LOGS_DIR, spec_folder, embed_type)
    os.makedirs(config_folder, exist_ok=True)
    os.makedirs(logs_folder, exist_ok=True)
    return config_folder


def write_config_file(config_folder, mode):
    base_config_path = os.path.join('data', 'original', 'base_%s.yml' % mode)
    to_write = open(base_config_path, 'r').read()
    if 'pre-trained' in config_folder and mode == 'train':  # or 'test' too?
        to_write += W2V_TEXT
    to_write = apply_parameters_to_config_file(to_write, config_folder)
    config_path = os.path.join(config_folder, '%s.yml' % mode)
    with open(config_path, 'w') as f: f.writelines(to_write)


def apply_parameters_to_config_file(to_write, config_folder):
    spec_folder, embed_type = os.path.split(config_folder)
    spec_folder = spec_folder.replace(CONFIG_DIR, '')[1:]  # no '/' at base
    n_steps = str(int(spec_folder.split('x')[-1]) * N_BASE_STEPS)  # TODO: check for -1 in test config file
    return to_write.replace('$SPEC_FOLDER', spec_folder)\
                   .replace('$EMBED_TYPE', embed_type)\
                   .replace('$N_STEPS', n_steps)


if __name__ == '__main__':
    main()
