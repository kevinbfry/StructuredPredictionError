from abc import ABCMeta, abstractmethod

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, RandomForestClassifier
from sklearn.utils.validation import check_is_fitted

from .tree import LinearSelector

class ParametricBaggingRegressor(LinearSelector, BaggingRegressor):

    def fit(
        self, 
        X, 
        y, 
        sample_weight=None, 
        chol_eps=None, 
        idx_tr=None, 
        do_param_boot=True
    ):
        self.do_param_boot = do_param_boot
        self.X_tr_ = X
        super().fit(
            X,
            y, 
            sample_weight=sample_weight, 
            chol_eps=chol_eps, 
            idx_tr=idx_tr, 
            do_param_boot=do_param_boot
        )
        return self

    def get_group_X(self, X, X_pred=None):
        check_is_fitted(self)

        return super().get_group_X(X)

    def get_linear_smoother(self, X, tr_idx, ts_idx, ret_full_P=False):
        assert(isinstance(self.base_estimator, LinearSelector))

        Ps = [est.get_linear_smoother(X, tr_idx, ts_idx, ret_full_P)[0] for est in self.estimators_]
        
        return Ps

class BlurredForest(RandomForestRegressor):
    def __init__(
        self,
        n_estimators=100,
        *,
        criterion="squared_error",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=1.0,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        ccp_alpha=0.0,
        max_samples=None,
    ):

        super().__init__(
            n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            ccp_alpha=ccp_alpha,
            max_samples=max_samples,
        )

    def fit(self, X, y, sample_weight=None, chol_eps=None, idx_tr=None, do_param_boot=True):
        self.do_param_boot = do_param_boot
        self.X_tr_ = X
        self.y_refit_ = y
        super().fit(X, y, sample_weight=sample_weight, chol_eps=chol_eps, idx_tr=idx_tr, do_param_boot=do_param_boot)
        if do_param_boot:
            if idx_tr is None:
                self.w_refit_ = [y + eps for eps in self.eps_]
            else:
                self.w_refit_ = [y + eps[idx_tr] for eps in self.eps_]

        return self

    def get_linear_smoother(self, X, tr_idx, ts_idx, Chol=None, ret_full_P=False):
    
        Xs = self.get_group_X(X)
        
        X_tss = [X[ts_idx,:] for X in Xs]
        X_trs = [X[tr_idx,:] for X in Xs]
        averaging_matrices = [X_tr.T / X_tr.T.sum(1)[:,None] for X_tr in X_trs]
        if ret_full_P:
            n = X.shape[0]
            full_averaging_matrices = [np.zeros((X_trs[0].shape[1],n)) for _ in range(len(averaging_matrices))]
            for full_averaging_matrix, averaging_matrix in zip(full_averaging_matrices, averaging_matrices):
                full_averaging_matrix[:,tr_idx] = averaging_matrix
        else:
            full_averaging_matrices = averaging_matrices
        return [X_ts @ averaging_matrix for X_ts, averaging_matrix in zip(X_tss, full_averaging_matrices)]

    def get_group_X(self, X):
        check_is_fitted(self)

        n = X.shape[0]
        leaf_node_array = self.apply(X)

        stored_n_leaves = hasattr(self, "n_leaves")
        if not stored_n_leaves:
            self.n_leaves = np.zeros(leaf_node_array.shape[1]).astype(int)
            self.leaves_map = []

        Gs = []
        for i in np.arange(leaf_node_array.shape[1]):
            leaf_nodes = leaf_node_array[:, i]
            _, indices = np.unique(leaf_nodes, return_inverse=True)

            n_leaves = np.amax(indices) + 1
            n = X.shape[0]

            G = np.zeros((n, n_leaves))
            G[np.arange(n), indices] = 1
            Gs.append(G)

        return Gs

    # def predict(self, X, tr_idx=None, ts_idx=None, full_refit=False, Chol=None):
    #     check_is_fitted(self)
    #     if full_refit or Chol is not None:
    #         assert(tr_idx is not None and ts_idx is not None)
    #         Ps = self.get_linear_smoother(X, tr_idx, ts_idx, Chol=Chol)#self.X_tr_, X, Chol=Chol)
    #         # ols_Ps = self.get_linear_smoother(self.X_tr_, X, Chol=None)
    #         # print("asdfsdfds")
    #         # assert(np.allclose(Ps, ols_Ps))
    #         # if self.bootstrap_type == "blur":
    #         if self.do_param_boot:
    #             if full_refit:
    #                 preds = [P @ self.y_refit_ for P in Ps]
    #             else:
    #                 preds = [P @ w for (P, w) in zip(Ps, self.w_refit_)]
    #         else:
    #             if full_refit:
    #                 preds = [P @ self.y_refit_ for P in Ps]
    #             else:
    #                 if Chol is None:
    #                     return self.predict(X)
    #                 else:
    #                     raise NotImplementedError("Not yet implemented for RF with GLS and W refit")
    #         pred = np.mean(preds, axis=0)
    #         return pred
    #     else:
    #         if ts_idx is not None:
    #             X_pred = X[ts_idx,:]
    #         else:
    #             X_pred = X
    #         return super().predict(X_pred)
            # return super().predict(X)#[ts_idx,:])


# class BlurredForestClassifier(RandomForestClassifier):
#     def __init__(
#         self,
#         n_estimators=100,
#         *,
#         criterion="gini",
#         max_depth=None,
#         min_samples_split=2,
#         min_samples_leaf=1,
#         min_weight_fraction_leaf=0.0,
#         max_features=1.0,
#         max_leaf_nodes=None,
#         min_impurity_decrease=0.0,
#         bootstrap=True,
#         oob_score=False,
#         n_jobs=None,
#         random_state=None,
#         verbose=0,
#         warm_start=False,
#         class_weight=None,
#         ccp_alpha=0.0,
#         max_samples=None,
#         bootstrap_type=None,
#     ):

#         super().__init__(
#             n_estimators,
#             criterion=criterion,
#             max_depth=max_depth,
#             min_samples_split=min_samples_split,
#             min_samples_leaf=min_samples_leaf,
#             min_weight_fraction_leaf=min_weight_fraction_leaf,
#             max_features=max_features,
#             max_leaf_nodes=max_leaf_nodes,
#             min_impurity_decrease=min_impurity_decrease,
#             bootstrap=bootstrap,
#             oob_score=oob_score,
#             n_jobs=n_jobs,
#             random_state=random_state,
#             verbose=verbose,
#             warm_start=warm_start,
#             ccp_alpha=ccp_alpha,
#             max_samples=max_samples,
#             # bootstrap_type=bootstrap_type,
#         )

#     def fit(self, X, y, sample_weight=None, chol_eps=None, idx_tr=None):
#         self.X_tr_ = X
#         self.y_refit_ = y
#         super().fit(X, y, sample_weight=sample_weight, chol_eps=chol_eps, idx_tr=idx_tr)
#         if self.bootstrap_type == "blur":
#             self.w_refit_ = [y + eps for eps in self.eps_]

#         return self

#     def get_linear_smoother(self, X, X_pred=None, Chol=None):
#         Gs = self.get_group_X(X)
#         if X_pred is None:
#             G_preds = Gs
#         else:
#             G_preds = self.get_group_X(X_pred)
#             delattr(self, "n_leaves")

#         # return [G_pred @ np.linalg.inv(G.T @ G) @ G.T for (G,G_pred) in zip(Gs, G_preds)]
#         if Chol is None:
#             return [G_pred @ np.linalg.pinv(G) for (G, G_pred) in zip(Gs, G_preds)]

#         return [
#             G_pred @ np.linalg.pinv(Chol.T @ G) @ Chol.T
#             for (G, G_pred) in zip(Gs, G_preds)
#         ]

#     # def get_group_X(self, X):
#     #   check_is_fitted(self)

#     #   leaf_node_array = self.apply(X)
#     #   n = X.shape[0]

#     #   Gs = []
#     #   for i in np.arange(leaf_node_array.shape[1]):
#     #       leaf_nodes = leaf_node_array[:,i]
#     #       _, indices = np.unique(leaf_nodes, return_inverse=True)

#     #       n_leaves = np.amax(indices) + 1

#     #       G = np.zeros((n, n_leaves))
#     #       G[np.arange(n),indices] = 1
#     #       Gs.append(G)

#     #   return Gs

#     def predict(self, X, full_refit=False, Chol=None):
#         if full_refit or Chol is not None:
#             Ps = self.get_linear_smoother(self.X_tr_, X, Chol=Chol)
#             # ols_Ps = self.get_linear_smoother(self.X_tr_, X, Chol=None)
#             # print("asdfsdfds")
#             # assert(np.allclose(Ps, ols_Ps))
#             if full_refit:
#                 preds = [P @ self.y_refit_ for P in Ps]
#             else:
#                 preds = [P @ w for (P, w) in zip(Ps, self.w_refit_)]
#             pred = np.mean(preds, axis=0)
#             return pred
#         else:
#             return super().predict(X)

#     def get_group_X(self, X):
#         check_is_fitted(self)

#         n = X.shape[0]
#         leaf_node_array = self.apply(X)

#         stored_n_leaves = hasattr(self, "n_leaves")
#         if not stored_n_leaves:
#             self.n_leaves = np.zeros(leaf_node_array.shape[1]).astype(int)
#             self.leaves_map = []

#         # stored_leaf_idx = hasattr(self, 'leaf_idx')
#         # if not stored_leaf_idx:
#         #   self.leaf_idx = []

#         Gs = []
#         for i in np.arange(leaf_node_array.shape[1]):
#             # tree = self.estimators_[i]
#             leaf_nodes = leaf_node_array[:, i]

#             vals, indices = np.unique(leaf_nodes, return_inverse=True)

#             # G = np.zeros((n, tree.tree_.node_count))
#             # G[np.arange(n), leaf_nodes] = 1

#             # if not stored_leaf_idx:
#             #   vals = np.unique(leaf_nodes)
#             #   G = G[:, vals]
#             #   self.leaf_idx.append(vals)
#             # else:
#             #   G = G[:, self.leaf_idx[i]]

#             if not stored_n_leaves:
#                 self.n_leaves[i] = n_leaves = np.amax(indices) + 1
#                 leaves_map = dict([(vals[i], i) for i in np.unique(indices)])
#                 assert n_leaves == len(leaves_map)
#                 self.leaves_map.append(leaves_map)
#             else:
#                 n_leaves = self.n_leaves[i]
#                 leaves_map = self.leaves_map[i]

#             # n_leaves = np.amax(indices) + 1

#             G = np.zeros((n, n_leaves))
#             leaf_idx = [leaves_map[l] for l in leaf_nodes]
#             G[np.arange(n), indices] = 1

#             Gs.append(G)

#         return Gs
