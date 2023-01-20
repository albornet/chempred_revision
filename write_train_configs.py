import os


CONFIGS_DIR = os.path.abspath('configs')
LOGS_DIR = os.path.abspath('logs')
DATA_DIR = os.path.abspath('data')
BASE_CONFIG_PATH = os.path.join(DATA_DIR, 'original', 'base_train.yml')
N_BASE_TRAIN_STEPS = 500000
N_BASE_VALID_STEPS = 10000
EMBED_TYPES = ['from-scratch', 'pre-trained']
W2V_TEXT = """
# Add pre-trained embeddings
both_embeddings: $DATA_FOLDER%sw2v-embeddings.txt
embeddings_type: word2vec
word_vec_size: 256
freeze_word_vecs_enc: True
freeze_word_vecs_dec: True
""" % os.path.sep
NOT_SHARE_VOCAB_TEXT = """tgt_vocab: $DATA_FOLDER$SEPtgt_vocab.vocab
share_vocab: False"""
SHARE_VOCAB_TEXT = """share_vocab: True"""


def main():
    for folder, subfolders, _ in os.walk(DATA_DIR):
        if len(subfolders) == 0:
            if 'x1' in folder and 'x10' not in folder:
                setup_train_configs(folder, EMBED_TYPES)
            elif 'original' not in folder:
                setup_train_configs(folder, [EMBED_TYPES[0]])
    print('Training configuration files generated!')


def setup_train_configs(data_folder, embed_types_for_this_data):
    spec_path = data_folder.split(DATA_DIR)[1][1:]
    for embed_type in embed_types_for_this_data:
        config_folder = os.path.join(CONFIGS_DIR, spec_path, embed_type)
        logs_folder = os.path.join(LOGS_DIR, spec_path, embed_type)
        os.makedirs(config_folder, exist_ok=True)
        os.makedirs(logs_folder, exist_ok=True)
        write_train_config_file(config_folder, data_folder, logs_folder)


def write_train_config_file(config_folder, data_folder, logs_folder):
    to_write = open(BASE_CONFIG_PATH, 'r').read()
    vocab_text = SHARE_VOCAB_TEXT
    if 'reagent-pred' in config_folder:
        vocab_text = NOT_SHARE_VOCAB_TEXT
    if 'pre-trained' in config_folder: to_write += W2V_TEXT
    fold_flag = os.path.basename(data_folder).split('x')[-1]
    n_steps_train_max = str(int(fold_flag) * N_BASE_TRAIN_STEPS)
    n_steps_for_valid = str(int(fold_flag) * N_BASE_VALID_STEPS)
    config_path = os.path.join(config_folder, 'train.yml')
    with open(config_path, 'w') as f:
        f.writelines(to_write.replace('$VOCAB_TEXT', vocab_text)
                             .replace('$LOGS_FOLDER', logs_folder)
                             .replace('$DATA_FOLDER', data_folder)
                             .replace('$N_STEPS_TRAIN_MAX', n_steps_train_max)
                             .replace('$N_STEPS_FOR_VALID', n_steps_for_valid)
                             .replace('$SEP', os.path.sep))


if __name__ == '__main__':
    main()
