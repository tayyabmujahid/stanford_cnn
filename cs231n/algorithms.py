import numpy as np


class NearestNeighbour:
    def __init__(self):
        pass

    def train(self, X, y):
        self.Xtr = X
        self.ytr = y

    def predict(self, X):
        num_test = X.shape[0]
        Ypred = np.zeros(num_test, dtype=self.ytr.dtype)

        for i in range(num_test):
            distances = np.sum(np.abs(self.Xtr - X[i, :]), axis=1)
            min_index = np.argmin(distances)
            Ypred[i] = self.ytr[min_index]

        return Ypred


class KNN:
    def __init__(self,k=3):

        self.k=k


    def train(self, X, y):
        self.Xtr = X
        self.ytr = y

    def predict(self, X):
        num_test = X.shape[0]
        Ypred = np.zeros(num_test, dtype=self.ytr.dtype)

        for i in range(num_test):
            distances = np.sum(np.abs(self.Xtr - X[i, :]), axis=1)

            sorted_indices = np.argpartition(distances,self.k)
            min_indices = sorted_indices[:self.k]
            Ypred_k = self.ytr[min_indices]

            unique, counts = np.unique(Ypred_k, return_counts=True)

            Ypred[i] = unique[np.argmax(counts)]


        return Ypred

nn = KNN(k=7)