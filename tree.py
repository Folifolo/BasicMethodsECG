from abc import ABC, abstractmethod

from method_utils import FdaUtils, LdaUtils
from utils import *


class TreeNode(ABC):
    @abstractmethod
    def __init__(self, thresh=5):
        self.method = None
        self.left_child = None
        self.right_child = None
        self.thresh = thresh

    @abstractmethod
    def grow(self):
        """Set left and right"""

    def initialise(self):
        if not self.is_have_childs():
            self.grow()

    def is_have_childs(self):
        return self.left_child is not None and self.right_child is not None

    def fit(self, x, y):
        self.method.fit(x, y)

        if self.is_have_childs():
            left, right = self.divide_data(x)
            if sum(y[left]) < self.thresh or sum(y[right]) > len(y[right]) - self.thresh:
                self.left_child = self.right_child = None
            else:
                self.right_child.fit(x[left], y[left])
                self.left_child.fit(x[right], y[right])

    def find_optimal_param(self, x, y):
        self.method.find_optimal_param(x, y)

        if self.is_have_childs():
            left, right = self.divide_data(x)
            if sum(y[left]) < self.thresh or sum(y[right]) > len(y[right]) - self.thresh:
                self.left_child = self.right_child = None
            else:
                self.right_child.find_optimal_param(x[left], y[left])
                self.left_child.find_optimal_param(x[right], y[right])

    def divide_data(self, x):
        probs = self.method.predict_proba(x)[:, 1]
        left = (probs <= self.method.m)
        right = (probs > self.method.m)
        return left, right

    def predict(self, x):
        if not self.is_have_childs():
            pred = self.method.predict(x)

        else:
            left, right = self.divide_data(x)
            l_pred = self.left_child.predict(x[left])
            r_pred = self.right_child.predict(x[right])
            pred = np.zeros(x.shape[0])
            pred[left] = l_pred
            pred[right] = r_pred

        return pred


class FdaTree(TreeNode):
    def __init__(self, threshold=5):
        super().__init__(threshold)
        self.method = FdaUtils()

    def grow(self):
        self.right_child = FdaTree()
        self.left_child = FdaTree()


class LdaTree(TreeNode):
    def __init__(self, threshold=5):
        super().__init__(threshold)
        self.method = LdaUtils()

    def grow(self):
        self.right_child = LdaTree()
        self.left_child = LdaTree()


if __name__ == "__main__":
    from sklearn.decomposition import PCA
    from validate import cross_validate_method
    from dataset import load_dataset_6002, MOST_FREQ_DIAGS_NUMS_OLD

    num_components = 100

    xy = load_dataset_6002('old')
    X = xy['x']
    Y = xy['y']
    pca = PCA(n_components=X.shape[0])
    b = pca.fit_transform(X)

    meth = FdaTree(3)
    for d in reversed(MOST_FREQ_DIAGS_NUMS_OLD):
        cross_validate_method(meth, b[:, :num_components], Y[:, d], 1)
