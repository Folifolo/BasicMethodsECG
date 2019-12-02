from abc import ABC, abstractmethod
from method_realisations import FisherDiscriminantAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

from utils import *


class MethodUtils(ABC):

    @abstractmethod
    def __init__(self):
        self.method = None

    def initialise(self):
        pass

    def fit(self, x, y):
        self.method.fit(x, y)
        self.find_optimal_param(x, y)

    def find_optimal_param(self, x, y):
        probs = self.method.predict_proba(x)[:, 1]

        y = [elem for _, elem in sorted(zip(probs, y))]
        y = np.array(y)
        probs.sort()
        se = []
        sp = []
        for p in range(len(probs)):
            tp = np.count_nonzero(y[p:] == 1)
            fp = np.count_nonzero(y[p:] == 0)
            tn = np.count_nonzero(y[:p] == 0)
            fn = np.count_nonzero(y[:p] == 1)
            se.append(tp / (tp + fn))
            sp.append(tn / (tn + fp))

        mx = np.argmax(-(1 - np.array(sp) - np.array(se)))

        self.m = probs[mx]

    def predict(self, x):
        if len(x) == 0:
            pass
        probs = self.method.predict_proba(x)[:, 1]

        return (probs > self.m).astype(int)

    def predict_proba(self, x):
        return self.method.predict_proba(x)


class LdaUtils(MethodUtils):
    def __init__(self):
        super().__init__()
        self.method = LinearDiscriminantAnalysis()


class FdaUtils(MethodUtils):
    def __init__(self):
        super().__init__()
        self.method = FisherDiscriminantAnalysis()


class LogRegUtils(MethodUtils):
    def __init__(self):
        super().__init__()
        self.method = LogisticRegression()

class QdaUtils(MethodUtils):
    def __init__(self):
        super().__init__()
        self.method = QuadraticDiscriminantAnalysis()
