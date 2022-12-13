import os
# started at 16:25


VOCAB_SCRIPT = 'python open-nmt/build_vocab.py'
TRAIN_SCRIPT = 'python open-nmt/train.py'
FOLDS = [1, 2, 5, 10, 20]


def main():
    compute_training_time()
    for fold in FOLDS:
        train_one_data_augmentation_level(fold)


def train_one_data_augmentation_level(fold):
    for folder, subfolders, _ in os.walk('./config'):
        if len(subfolders) == 0\
        and str(fold) in folder and str(10 * fold) not in folder:
            config = os.path.join(folder, 'train.yml')
            os.system('%s -config %s -n_sample -1' % (VOCAB_SCRIPT, config))
            os.system('%s -config %s' % (TRAIN_SCRIPT, config))


def compute_training_time():
    single_gpu_seconds_per_step = 20 / 100
    max_steps_per_fold = 500000
    n_folds_per_task_exp_1 = sum(FOLDS)  # data augmentation
    n_folds_per_task_exp_2 = 2 * 2 * 2 - 1  # all input schemes, minus run done in exp_1
    n_tasks_exp_1 = 5
    n_tasks_exp_2 = 3
    n_folds_exp_1 = n_folds_per_task_exp_1 * n_tasks_exp_1
    n_folds_exp_2 = n_folds_per_task_exp_2 * n_tasks_exp_2
    n_seconds_per_day = 60 * 60 * 24
    n_steps_exp_1 = n_folds_exp_1 * max_steps_per_fold
    n_steps_exp_2 = n_folds_exp_2 * max_steps_per_fold
    single_gpu_seconds_exp_1 = n_steps_exp_1 * single_gpu_seconds_per_step
    single_gpu_seconds_exp_2 = n_steps_exp_2 * single_gpu_seconds_per_step
    single_gpu_days_exp_1 = single_gpu_seconds_exp_1 / n_seconds_per_day
    single_gpu_days_exp_2 = single_gpu_seconds_exp_2 / n_seconds_per_day
    print('-- Exp. 1 takes at most %.2f days on a single gpu' % single_gpu_days_exp_1)
    print('-- Exp. 2 takes at most %.2f days on a single gpu' % single_gpu_days_exp_2)


if __name__ == '__main__':
    main()
