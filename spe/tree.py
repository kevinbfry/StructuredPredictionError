from abc import ABC, abstractmethod

import numpy as np

from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.validation import check_is_fitted


class LinearSelector(ABC):
    @abstractmethod
    def get_linear_smoother(self, X):
        pass

    @abstractmethod
    def get_group_X(self, X, tr_idx, ts_idx, ret_full_P=False):
        pass

    def predict(
        self,
        X,
        tr_idx=None,
        ts_idx=None,
        y_refit=None,
    ):
        check_is_fitted(self)
        if tr_idx is None and ts_idx is None and y_refit is None:
            return super().predict(X)
        elif tr_idx is not None and ts_idx is not None and y_refit is not None:
            Ps = self.get_linear_smoother(X, tr_idx, ts_idx)
            preds = [P @ y_refit for P in Ps]
            pred = np.mean(preds, axis=0)
            return pred
        else:
            raise ValueError("Either all of 'tr_idx', 'ts_idx', 'y_refit' must be None or all must not be None")


class Tree(LinearSelector, DecisionTreeRegressor):

    def get_group_X(self, X):
        check_is_fitted(self)

        leaf_nodes = self.apply(X)
        _, indices = np.unique(leaf_nodes, return_inverse=True)

        n_leaves = np.amax(indices) + 1
        n = X.shape[0]
        
        G = np.zeros((n, n_leaves))

        G[np.arange(n), indices] = 1

        return G

    def get_linear_smoother(self, X, tr_idx, ts_idx, ret_full_P=False):
        X = self.get_group_X(X)
        
        X_ts = X[ts_idx,:]
        X_tr = X[tr_idx,:]
        averaging_matrix = X_tr.T / X_tr.T.sum(1)[:,None]
        
        if ret_full_P:
            n = X.shape[0]
            full_averaging_matrix = np.zeros((X_tr.shape[1],n))
            full_averaging_matrix[:,tr_idx] = averaging_matrix
            return [X_ts @ full_averaging_matrix]
        return [X_ts @ averaging_matrix]
