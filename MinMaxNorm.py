import numpy as np


np.random.seed(1337)  # for reproducibility


class MinMaxNormalization(object):
    """
        MinMax Normalization --> [-1, 1]
        x = (x - min) / (max - min).
        x = x * 2 - 1
    """

    def __init__(self):
        pass

    def fit(self, X_train, X_test, Y_train, Y_test):

        """
        Fit the data to the normalizer, get maximum and minimum
        """

        self._min = min(np.min(X_train), np.min(X_test), np.min(Y_train), np.min(Y_test))
        self._max = max(np.max(X_train), np.max(X_test), np.max(Y_train), np.max(Y_test))

        

        print("min:", self._min, "max:", self._max)

    def transform(self, X):

        """
        Normalize the data
        """

        X = 1. * (X - self._min) / (self._max - self._min)
        X = X * 2. - 1. #to put the value between -1 and 1
        return X

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):

        """
        De-Normalize the data
        """

        X = (X + 1.) / 2.
        X = 1. * X * (self._max - self._min) + self._min
        return X

