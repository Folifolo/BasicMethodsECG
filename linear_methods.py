import numpy as np

from abc import ABC, abstractmethod

class LinearMethod(ABC):

    def __init__(self):
        pass

    def fit(self, x, y):
        means = []
        cov = []
        self.priors_ = np.bincount(y.astype(int)) / float(len(y))
        for ind in [0, 1]:
            x_current = x[y == ind, :]
            mean_current = x_current.mean(axis=0)
            means.append(mean_current)

            cov.append(np.cov(x_current.T))

        self.means_ = np.asarray(means)
        self.covariations_ = np.asarray(cov)

    @abstractmethod
    def _predict(self, x):
        pass

    @abstractmethod
    def predict_proba(self, x):
        pass

class LinearDiscriminantAnalysis(LinearMethod):
    def _predict(self, x):
        x = np.array(x)
        res = [0,0]
        for ind in [0, 1]:
            cov_inv = np.linalg.inv(self.covariations_[ind])
            mu_cov_x = np.dot(np.dot(self.means_[ind], cov_inv), x.T)
            mu_cov_mu = np.dot(np.dot(self.means_[ind], cov_inv), self.means_[ind].T)

            res[ind] = mu_cov_x - 0.5 * mu_cov_mu + np.log(self.priors_[ind])

        res = np.array(res)
        return res


    def predict(self, X):
        return self._predict(X).argmax(0)

    def predict_proba(self, x):
        values = self._predict(x)
        values = np.swapaxes(values, 1,0)
        likelihood = np.exp(values - values.max(axis=1)[:, np.newaxis])
        return likelihood / likelihood.sum(axis=1)[:, np.newaxis]

class FisherDiscriminantAnalysis(LinearMethod):
    def fit(self, x, y):
        super().fit(x, y)
        cov_sum = np.linalg.inv(self.covariations_[0] + self.covariations_[1])
        mean_dif = (self.means_[1] - self.means_[0])
        self.w = np.dot(cov_sum, mean_dif)

    def _predict(self, x):

        x = np.array(x)
        res = np.dot(x, self.w)
        return res

    def predict(self, X, c):
        return (self._predict(X) > c).astype(int)

    def predict_proba(self, x):
        values = self._predict(x)
        values = np.expand_dims(values, 1)
        values = np.concatenate((-values, values), 1)
        return values
