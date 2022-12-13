import os


TEST_SCRIPT = 'python open-nmt/translate.py'
FOLDS = [1, 2, 5, 10, 20]


def main():
    for fold in FOLDS:
        train_one_data_augmentation_level(fold)


def train_one_data_augmentation_level(fold):
    for folder, subfolders, _ in os.walk('./config'):
        if len(subfolders) == 0\
        and str(fold) in folder and str(10 * fold) not in folder:
            config = os.path.join(folder, 'test.yml')
            os.system('%s -config %s' % (TEST_SCRIPT, config))


if __name__ == '__main__':
    main()
