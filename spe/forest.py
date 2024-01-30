import numpy as np

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