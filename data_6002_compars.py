import numpy as np
from matplotlib import pyplot as plt

from dataset import load_dataset_6002

if __name__ == "__main__":
    xy = load_dataset_6002('new')
    X1 = xy['x']
    xy = load_dataset_6002('old_new')
    X2 = xy['x']

    meanX = np.mean(X1, axis=0)
    stdX = np.std(X1, axis=0)
    maxX = np.max(X1, axis=0)
    minX = np.min(X1, axis=0)

    plt.subplot(211)
    plt.plot(meanX)
    plt.plot(stdX)
    plt.plot(maxX)
    plt.plot(minX)

    meanX1 = np.mean(X2, axis=0)
    stdX1 = np.std(X2, axis=0)
    maxX1 = np.max(X2, axis=0)
    minX1 = np.min(X2, axis=0)

    plt.subplot(212)
    plt.plot(meanX1)
    plt.plot(stdX1)
    plt.plot(maxX1)
    plt.plot(minX1)
    plt.legend(('mean', 'var', 'max', 'min'))
    plt.show()
