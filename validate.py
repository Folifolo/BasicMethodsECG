import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from dataset import load_dataset_6002
from utils import cross_val


def get_se_sp(y, y_prediction):
    tn, fp, fn, tp = confusion_matrix(y, y_prediction).ravel()
    se = tp / (tp + fn)
    sp = tn / (tn + fp)
    return se, sp


def cross_validate_method(method, x, y, test_part_size=1):
    y_prediction = np.empty(x.shape[0])

    for train_index, test_index in cross_val(x.shape[0], test_part_size):
        method.initialise()
        method.fit(x[train_index], y[train_index])

        y_prediction[test_index] = method.predict(x[test_index])

    se, sp = get_se_sp(y, y_prediction)
    print("Val. Se = %s, Val. Sp = %s" % (round(sp, 4), round(se, 4)))

    return se, sp


def test_on_two_sets(method, diagnosis_num, principal_comp_num=100):
    old_dataset = load_dataset_6002('old_new')
    new_dataset = load_dataset_6002('new')
    x_test = [None] * 2
    y_test = [None] * 2

    x_test[1] = new_dataset['x']
    y_test[1] = new_dataset['y']

    x_train, x_test[0], y_train, y_test[0] = train_test_split(old_dataset['x'], old_dataset['y'], test_size=0.25,
                                                              random_state=42)

    pca = PCA(n_components=principal_comp_num)
    x_train = pca.fit_transform(x_train)

    method.initialise()
    method.fit(x_train, y_train[:, diagnosis_num])
    se = [0] * 2
    sp = [0] * 2
    for idx in [0, 1]:
        x_test[idx] = pca.transform(x_test[idx])
        y_test[idx] = y_test[idx][:, diagnosis_num]

        y_prediction = method.predict(x_test[idx])
        se[idx], sp[idx] = get_se_sp(y_test[idx], y_prediction)

    print("Val.1 Se = %s, Val.1 Sp = %s, "
          "Val.2 Se = %s, Val.2 Sp = %s" % (round(sp[0], 4), round(se[0], 4), round(sp[1], 4), round(se[1], 4)))


def validate_and_test(method, diagnosis_num, principal_comp_num=100):
    old_dataset = load_dataset_6002('old_new')
    new_dataset = load_dataset_6002('new')
    x_test = [None] * 2
    y_test = [None] * 2

    x_test[1] = new_dataset['x']
    y_test[1] = new_dataset['y']

    x_train, x_test[0], y_train, y_test[0] = train_test_split(old_dataset['x'], old_dataset['y'], test_size=0.25,
                                                              random_state=42)

    pca = PCA(n_components=principal_comp_num)
    x_train = pca.fit_transform(x_train)

    method.initialise()
    method.fit(x_train, y_train[:, diagnosis_num])
    se = [0] * 2
    sp = [0] * 2
    for idx in [0, 1]:
        x_test[idx] = pca.transform(x_test[idx])
        y_test[idx] = y_test[idx][:, diagnosis_num]

    method.find_optimal_param(x_test[0], y_test[0])

    for idx in [0, 1]:
        y_prediction = method.predict(x_test[idx])
        se[idx], sp[idx] = get_se_sp(y_test[idx], y_prediction)

    print("Val. Se = %s, Val. Sp = %s, "
          "Tests Se = %s, Test Sp = %s" % (round(sp[0], 4), round(se[0], 4), round(sp[1], 4), round(se[1], 4)))
