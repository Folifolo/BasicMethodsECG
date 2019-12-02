import numpy as np

from os import listdir
from os.path import isfile, join
from dataset import load_dataset_6002, RANDOM_STATE, SPLIT_SIZE
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    old_dataset = load_dataset_6002('old_new')
    new_dataset = load_dataset_6002('new')

    xt2 = new_dataset['x']

    xtr, xt1 = train_test_split(old_dataset['x'], test_size=SPLIT_SIZE,
                                random_state=RANDOM_STATE)

    train_size = len(xtr)
    test1_size = len(xt1)
    test2_size = len(xt2)

    LOG_FOLDER = "logs\\"
    table_names = [f for f in listdir(LOG_FOLDER) if isfile(join(LOG_FOLDER, f))]
    best_result = np.zeros(15)
    best_method = np.array([''] * 15, dtype=np.dtype('U256'))
    best_se_sp = np.zeros((15, 2))
    for name in table_names:
        if (name[0] == "v"):
            continue
        elif (name[0] == "t"):

            table_test = np.genfromtxt(LOG_FOLDER + name, delimiter=';')
            name = name[5:]
            test1 = table_test[:, :2]
            test2 = table_test[:, 2:]

            table_val = np.genfromtxt(LOG_FOLDER + 'val_' + name, delimiter=';')
            val = table_val[:, :2]
            test_val = table_val[:, 2:]

            denum = (2 * (test1_size + test2_size))
            mean_se_sp = ((test1 + val) * test1_size + (test2 + test_val) * test2_size) / denum
            mean_sum = (np.sum((test1 + val), axis=1) * test1_size
                      + np.sum((test2 + test_val), axis=1) * test2_size) / denum
            better_nums = best_result < mean_sum
            best_result[better_nums] = mean_sum[better_nums]
            best_se_sp[better_nums] = mean_se_sp[better_nums]
            tmp = np.array([name] * np.sum(better_nums))
            best_method[better_nums] = tmp

    print(best_method)
    print(best_se_sp)
    print(1)
