import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-t', '--test', action='store_true')
parser.add_argument('-r', '--roundtrip', action='store_true')
args = parser.parse_args()


TRANSLATE_SCRIPT = 'python open-nmt/translate.py'
CONFIGS_DIR = os.path.join('.', 'configs')
MODES = ['test', 'test-50k', 'roundtrip', 'roundtrip-50k']
FOLDS = [1, 2, 5, 10, 20]
DO_TEST = args.test
DO_ROUNDTRIP = args.roundtrip


def main():
    if DO_TEST: modes_to_run = [m for m in MODES if 'test' in m]
    if DO_ROUNDTRIP: modes_to_run = [m for m in MODES if 'roundtrip' in m]
    if not DO_TEST and not DO_ROUNDTRIP:
        raise ValueError('Use this script with one of the following argument:'\
                         '\n-t to generate predictions for test tasks'\
                         '\n-r to generate predictions for roundtrip tasks '\
                         '(only after tests are run!)')
    if DO_TEST and DO_ROUNDTRIP:
        raise ValueError('You should use this script with only one argument.'\
                         '\n-t to generate predictions for test tasks'\
                         '\n-r to generate prediction for roundtrip tasks'\
                         '\nIt is first required to run test predictions,'\
                         '\nthen write roundtrip config scripts,'\
                         '\nand only then generate roundtrip predictions.')
    for mode in modes_to_run:
        generate_predictions_with_all_models(mode=mode)


def generate_predictions_with_all_models(mode):
    for folder, _, files in os.walk(CONFIGS_DIR):
        if '%s.yml' % mode in files:
            config_path = os.path.join(folder, '%s.yml' % mode)
            print('Starting %s' % config_path)
            os.system('%s -config %s' % (TRANSLATE_SCRIPT, config_path))
            print('Done %s' % config_path)


if __name__ == '__main__':
    main()
