from warnings import catch_warnings, simplefilter, warn

import numpy as np

from scipy.sparse import issparse

from sklearn.exceptions import DataConversionWarning
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import _fit_context
from sklearn.tree._tree import DOUBLE, DTYPE
from sklearn.utils import check_random_state, compute_sample_weight
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.parallel import Parallel, delayed
from sklearn.ensemble._forest import (
    _generate_sample_indices, 
    _get_n_samples_bootstrap
)
from sklearn.utils.validation import (
    _check_sample_weight,
    check_is_fitted,
)


MAX_INT = np.iinfo(np.int32).max


def _parallel_build_trees(
    tree,
    bootstrap,
    X,
    y,
    sample_weight,
    tree_idx,
    n_trees,
    verbose=0,
    class_weight=None,
    n_samples_bootstrap=None,
    missing_values_in_feature_mask=None,
    ### KF:
    chol_eps=None,
    do_param_boot=False,
    ###
):
    """
    Private function used to fit a single tree in parallel."""
    if verbose > 1:
        print("building tree %d of %d" % (tree_idx + 1, n_trees))

    ### KF:
    if bootstrap:
        n_samples = X.shape[0]
        if sample_weight is None:
            curr_sample_weight = np.ones((n_samples,), dtype=np.float64)
        else:
            curr_sample_weight = sample_weight.copy()

        # if bootstrap_type == 'blur':
        if do_param_boot:
            n_samples = X.shape[0]
            # if sample_weight is None:
            #     curr_sample_weight = np.ones((n_samples,), dtype=np.float64)
            # else:
            #     curr_sample_weight = sample_weight.copy()

            if chol_eps is None:
                eps = np.random.randn(n_samples,1)
            else:
                eps = np.random.randn(chol_eps.shape[0],1)
                eps = chol_eps @ eps

            ## TODO: shouldn't need this... should just update chol_eps before passing into this funciton...


            # w = y + eps_tr
            w = y + eps


            tree.fit(X, w, sample_weight=curr_sample_weight, check_input=True)
            return tree, eps

        indices = _generate_sample_indices(
            tree.random_state, n_samples, n_samples_bootstrap
        )
        sample_counts = np.bincount(indices, minlength=n_samples)
        curr_sample_weight *= sample_counts

        if class_weight == "subsample":
            with catch_warnings():
                simplefilter("ignore", DeprecationWarning)
                curr_sample_weight *= compute_sample_weight("auto", y, indices=indices)
        elif class_weight == "balanced_subsample":
            curr_sample_weight *= compute_sample_weight("balanced", y, indices=indices)

        tree._fit(
            X,
            y,
            sample_weight=curr_sample_weight,
            check_input=False,
            missing_values_in_feature_mask=missing_values_in_feature_mask,
        )

        return tree
    else:
        tree._fit(
            X,
            y,
            sample_weight=sample_weight,
            check_input=False,
            missing_values_in_feature_mask=missing_values_in_feature_mask,
        )

        return tree
    ###
    


class BlurredForestRegressor(RandomForestRegressor):
    # def __init__(
    #     self,
    #     n_estimators=100,
    #     *,
    #     criterion="squared_error",
    #     max_depth=None,
    #     min_samples_split=2,
    #     min_samples_leaf=1,
    #     min_weight_fraction_leaf=0.0,
    #     max_features=1.0,
    #     max_leaf_nodes=None,
    #     min_impurity_decrease=0.0,
    #     bootstrap=True,
    #     oob_score=False,
    #     n_jobs=None,
    #     random_state=None,
    #     verbose=0,
    #     warm_start=False,
    #     ccp_alpha=0.0,
    #     max_samples=None,
    #     monotonic_cst=None,
    # ):

    #     super().__init__(
    #         n_estimators,
    #         criterion=criterion,
    #         max_depth=max_depth,
    #         min_samples_split=min_samples_split,
    #         min_samples_leaf=min_samples_leaf,
    #         min_weight_fraction_leaf=min_weight_fraction_leaf,
    #         max_features=max_features,
    #         max_leaf_nodes=max_leaf_nodes,
    #         min_impurity_decrease=min_impurity_decrease,
    #         bootstrap=bootstrap,
    #         oob_score=oob_score,
    #         n_jobs=n_jobs,
    #         random_state=random_state,
    #         verbose=verbose,
    #         warm_start=warm_start,
    #         ccp_alpha=ccp_alpha,
    #         max_samples=max_samples,
    #         monotonic_cst=monotonic_cst,
    #     )
        
    @_fit_context(prefer_skip_nested_validation=True)
    ## KF:
    def fit(
        self, 
        X, 
        y, 
        sample_weight=None, 
        chol_eps=None, 
        do_param_boot=False,
    ):
        """
        Build a forest of trees from the training set (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Internally, its dtype will be converted
            to ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csc_matrix``.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node. In the case of
            classification, splits are also ignored if they would result in any
            single class carrying a negative weight in either child node.

        chol_eps : array-like of shape (n_samples,n_samples), optional
            Cholesky of parametric bootstrap covariance matrix. In the case of 
            ``do_param_boot`` is ``False``, ``chol_eps`` is ignored. If 
            ``chol_eps`` is ``None`` and ``do_param_boot`` is ``True``, then 
            ``chol_eps`` is ``np.eye(n_samples)``. Default is ``None``.

        do_param_boot : bool, optional
            If ``True`` performs parametric bootstrap sampling. Default is ``False``.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        # Validate or convert input data
        if issparse(y):
            raise ValueError("sparse multilabel-indicator for y is not supported.")

        X, y = self._validate_data(
            X,
            y,
            multi_output=True,
            accept_sparse="csc",
            dtype=DTYPE,
            force_all_finite=False,
        )
        # _compute_missing_values_in_feature_mask checks if X has missing values and
        # will raise an error if the underlying tree base estimator can't handle missing
        # values. Only the criterion is required to determine if the tree supports
        # missing values.
        estimator = type(self.estimator)(criterion=self.criterion)
        missing_values_in_feature_mask = (
            estimator._compute_missing_values_in_feature_mask(
                X, estimator_name=self.__class__.__name__
            )
        )

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X)

        if issparse(X):
            # Pre-sort indices to avoid that each individual tree of the
            # ensemble sorts the indices.
            X.sort_indices()

        y = np.atleast_1d(y)
        if y.ndim == 2 and y.shape[1] == 1:
            warn(
                (
                    "A column-vector y was passed when a 1d array was"
                    " expected. Please change the shape of y to "
                    "(n_samples,), for example using ravel()."
                ),
                DataConversionWarning,
                stacklevel=2,
            )

        if y.ndim == 1:
            # reshape is necessary to preserve the data contiguity against vs
            # [:, np.newaxis] that does not.
            y = np.reshape(y, (-1, 1))

        if self.criterion == "poisson":
            if np.any(y < 0):
                raise ValueError(
                    "Some value(s) of y are negative which is "
                    "not allowed for Poisson regression."
                )
            if np.sum(y) <= 0:
                raise ValueError(
                    "Sum of y is not strictly positive which "
                    "is necessary for Poisson regression."
                )

        self._n_samples, self.n_outputs_ = y.shape

        y, expanded_class_weight = self._validate_y_class_weight(y)

        if getattr(y, "dtype", None) != DOUBLE or not y.flags.contiguous:
            y = np.ascontiguousarray(y, dtype=DOUBLE)

        if expanded_class_weight is not None:
            if sample_weight is not None:
                sample_weight = sample_weight * expanded_class_weight
            else:
                sample_weight = expanded_class_weight

        if not self.bootstrap and self.max_samples is not None:
            raise ValueError(
                "`max_sample` cannot be set if `bootstrap=False`. "
                "Either switch to `bootstrap=True` or set "
                "`max_sample=None`."
            )
        elif self.bootstrap:
            n_samples_bootstrap = _get_n_samples_bootstrap(
                n_samples=X.shape[0], max_samples=self.max_samples
            )
        else:
            n_samples_bootstrap = None

        self._n_samples_bootstrap = n_samples_bootstrap

        self._validate_estimator()

        if not self.bootstrap and self.oob_score:
            raise ValueError("Out of bag estimation only available if bootstrap=True")

        random_state = check_random_state(self.random_state)

        if not self.warm_start or not hasattr(self, "estimators_"):
            # Free allocated memory, if any
            self.estimators_ = []

        ### KF:
        # self.chol_eps = chol_eps

        # if self.bootstrap_type == 'blur' and not hasattr(self, "eps_"):
        if do_param_boot and not hasattr(self, "eps_"):
            self.eps_ = []
        ###

        n_more_estimators = self.n_estimators - len(self.estimators_)

        if n_more_estimators < 0:
            raise ValueError(
                "n_estimators=%d must be larger or equal to "
                "len(estimators_)=%d when warm_start==True"
                % (self.n_estimators, len(self.estimators_))
            )

        elif n_more_estimators == 0:
            warn(
                "Warm-start fitting without increasing n_estimators does not "
                "fit new trees."
            )
        else:
            if self.warm_start and len(self.estimators_) > 0:
                # We draw from the random state to get the random state we
                # would have got if we hadn't used a warm_start.
                random_state.randint(MAX_INT, size=len(self.estimators_))

            trees = [
                self._make_estimator(append=False, random_state=random_state)
                for i in range(n_more_estimators)
            ]

            # Parallel loop: we prefer the threading backend as the Cython code
            # for fitting the trees is internally releasing the Python GIL
            # making threading more efficient than multiprocessing in
            # that case. However, for joblib 0.12+ we respect any
            # parallel_backend contexts set at a higher level,
            # since correctness does not rely on using threads.
            trees = Parallel(
                n_jobs=self.n_jobs,
                verbose=self.verbose,
                prefer="threads",
            )(
                delayed(_parallel_build_trees)(
                    t,
                    self.bootstrap,
                    X,
                    y,
                    sample_weight,
                    i,
                    len(trees),
                    verbose=self.verbose,
                    class_weight=self.class_weight,
                    n_samples_bootstrap=n_samples_bootstrap,
                    missing_values_in_feature_mask=missing_values_in_feature_mask,
                    ### KF:
                    chol_eps=chol_eps,#self.chol_eps,
                    do_param_boot=do_param_boot,
                    ###
                )
                for i, t in enumerate(trees)
            )

            ### KF:
            # if self.bootstrap_type == 'blur':
            if do_param_boot:
                trees, eps = zip(*trees)
                self.eps_.extend(eps)
            # else:
            #     trees, boot_weights = zip(*trees)
            #     self.boot_weights_.extend(boot_weights)
            ###

            # Collect newly grown trees
            self.estimators_.extend(trees)

        if self.oob_score and (
            n_more_estimators > 0 or not hasattr(self, "oob_score_")
        ):
            y_type = type_of_target(y)
            if y_type == "unknown" or (
                self._estimator_type == "classifier"
                and y_type == "multiclass-multioutput"
            ):
                # FIXME: we could consider to support multiclass-multioutput if
                # we introduce or reuse a constructor parameter (e.g.
                # oob_score) allowing our user to pass a callable defining the
                # scoring strategy on OOB sample.
                raise ValueError(
                    "The type of target cannot be used to compute OOB "
                    f"estimates. Got {y_type} while only the following are "
                    "supported: continuous, continuous-multioutput, binary, "
                    "multiclass, multilabel-indicator."
                )

            if callable(self.oob_score):
                self._set_oob_score_and_attributes(
                    X, y, scoring_function=self.oob_score
                )
            else:
                self._set_oob_score_and_attributes(X, y)

        # Decapsulate classes_ attributes
        if hasattr(self, "classes_") and self.n_outputs_ == 1:
            self.n_classes_ = self.n_classes_[0]
            self.classes_ = self.classes_[0]

        if do_param_boot:
            self.w_refit_ = [y + eps for eps in self.eps_]


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