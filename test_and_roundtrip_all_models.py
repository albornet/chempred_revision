import os


TEST_SCRIPT = 'python open-nmt/translate.py'
CONFIGS_DIR = os.path.join('.', 'configs')
# MODES = ['test', 'test-50k', 'roundtrip', 'roundtrip-50k']
MODES = ['roundtrip', 'roundtrip-50k']
FOLDS = [1, 2, 5, 10, 20]


def main():
    for mode in MODES:
        generate_predictions_with_all_models(mode=mode)


def generate_predictions_with_all_models(mode):
    for folder, _, files in os.walk(CONFIGS_DIR):
        if '%s.yml' % mode in files:
            config_path = os.path.join(folder, '%s.yml' % mode)
            print('Starting %s' % config_path)
            os.system('%s -config %s' % (TEST_SCRIPT, config_path))
            print('Done %s' % config_path)


if __name__ == '__main__':
    main()
