from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.metrics import confusion_matrix

from utils import cross_val


def cross_validate_method(method, x, y, test_part_size=1):
    y_prediction = np.empty(x.shape[0])

    for train_index, test_index in cross_val(x.shape[0], test_part_size):
        method.initialise()
        method.fit(x[train_index], y[train_index])

        y_prediction[test_index] = method.predict(x[test_index])

    tn, fp, fn, tp = confusion_matrix(y, y_prediction).ravel()

    se = tp / (tp + fn)
    sp = tn / (tn + fp)
    print("Val. Se = %s, Val. Sp = %s" % (round(sp, 4), round(se, 4)))

    return se, sp
