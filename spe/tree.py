from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.validation import check_is_fitted


class LinearSelector(ABC):
    @abstractmethod
    def get_linear_smoother(self, X):
        pass

    @abstractmethod
    def get_group_X(self, X, tr_idx, ts_idx, ret_full_P=False):
        pass


class Tree(LinearSelector, DecisionTreeRegressor):
    # def get_linear_smoother(self, X):
    #     G = self.get_group_X(X)
    #     # return G @ np.linalg.inv(G.T @ G) @ G.T
    #     return G @ np.linalg.pinv(G)

    def get_group_X(self, X):#, is_train):
        check_is_fitted(self)

        leaf_nodes = self.apply(X)
        _, indices = np.unique(leaf_nodes, return_inverse=True)

        n_leaves = np.amax(indices) + 1
        n = X.shape[0]
        # print("X shape", X.shape, "n leaves", n_leaves)

        # if is_train:
        G = np.zeros((n, n_leaves))
        #     self.n_leaves = n_leaves
        # else:
        #     G = np.zeros((n, self.n_leaves))

        G[np.arange(n), indices] = 1

        return G

    def get_linear_smoother(self, X, tr_idx, ts_idx, ret_full_P=False):#, X_pred=None):
        X = self.get_group_X(X)#, is_train=True)
        # if X_pred is None:
        #     X_pred = X
        # else:
        #     X_pred = self.get_group_X(X_pred, is_train=False)

        # print("X", X, "\n", "pinv", np.linalg.pinv(X))
        X_ts = X[ts_idx,:]
        X_tr = X[tr_idx,:]
        averaging_matrix = X_tr.T / X_tr.T.sum(1)[:,None]
        # print(X, averaging_matrix)
        # assert(0==1)
        if ret_full_P:
            n = X.shape[0]
            full_averaging_matrix = np.zeros((X_tr.shape[1],n))
            full_averaging_matrix[:,tr_idx] = averaging_matrix
            return X_ts @ full_averaging_matrix
        return X_ts @ averaging_matrix
