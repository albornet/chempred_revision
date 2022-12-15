import os


TEST_SCRIPT = 'python open-nmt/translate.py'
CONFIG_DIR = os.path.join('.', 'config')
FOLDS = [1, 2, 5, 10, 20]


def main():
    for folder, subfolders, files in os.walk(CONFIG_DIR):
        if len(subfolders) == 0:
            for file in files:  # ['test.yml'] or ['test.yml', 'config.yml']
                config = os.path.join(folder, file)
                os.system('%s -config %s' % (TEST_SCRIPT, config))


if __name__ == '__main__':
    main()
