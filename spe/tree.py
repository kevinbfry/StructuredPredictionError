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
    def get_group_X(self, X):
        pass


class Tree(LinearSelector, DecisionTreeRegressor):
    def get_linear_smoother(self, X):
        G = self.get_group_X(X)
        # return G @ np.linalg.inv(G.T @ G) @ G.T
        return G @ np.linalg.pinv(G)

    def get_group_X(self, X):
        check_is_fitted(self)

        leaf_nodes = self.apply(X)
        _, indices = np.unique(leaf_nodes, return_inverse=True)

        n_leaves = np.amax(indices) + 1
        n = X.shape[0]

        G = np.zeros((n, n_leaves))
        G[np.arange(n), indices] = 1

        return G
