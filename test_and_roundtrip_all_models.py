import os


TEST_SCRIPT = 'python open-nmt/translate.py'
CONFIG_DIR = os.path.join('.', 'config')
FOLDS = [1, 2, 5, 10, 20]


def main():
    # First generate predictions with all possible models
    generate_predictions_with_all_models(mode='test')
    # Then only uses these predictions for the round-trip experiment
    generate_predictions_with_all_models(mode='roundtrip')


def generate_predictions_with_all_models(mode):
    for folder, _, files in os.walk(CONFIG_DIR):
        if '%s.yml' % mode in files:
            config_path = os.path.join(folder, '%s.yml' % mode)
            os.system('%s -config %s' % (TEST_SCRIPT, config_path))
            print('Done %s' % config_path)


if __name__ == '__main__':
    main()
