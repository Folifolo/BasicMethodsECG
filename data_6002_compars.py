import pickle as pkl

import numpy as np
from matplotlib import pyplot as plt
from dataset import load_dataset_6002

if __name__ == "__main__":
    path = 'C:\\Users\\donte_000\\PycharmProjects\\ClassificationECG\\data\\6002_old_NN.pkl'

    xy = load_dataset_6002('new')

    X = xy['x']
    print(X.shape)
    meanX = np.mean(X, axis=0)
    stdX = np.std(X, axis=0)
    maxX = np.max(X, axis=0)
    minX = np.min(X, axis=0)


    plt.plot(meanX)
    plt.plot(stdX)
    plt.plot(maxX)
    plt.plot(minX)
    plt.legend(('mean', 'var', 'max', 'min'))
    plt.show()
