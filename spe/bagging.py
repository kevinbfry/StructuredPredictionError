from functools import partial
import itertools
import numbers
from warnings import warn

import numpy as np

from scipy.sparse import issparse

from sklearn.ensemble import BaggingRegressor
from sklearn.utils import check_random_state, indices_to_mask
from sklearn.utils.metadata_routing import (
    _raise_for_unsupported_routing,
)
from sklearn.utils.parallel import Parallel, delayed
from sklearn.utils.random import sample_without_replacement
from sklearn.utils.validation import (
    _check_sample_weight,
    check_is_fitted,
    has_fit_parameter
)
from sklearn.ensemble._base import _partition_estimators

from .tree import LinearSelector

MAX_INT = np.iinfo(np.int32).max


def _generate_indices(random_state, bootstrap, n_population, n_samples):
    """Draw randomly sampled indices."""
    # Draw sample indices
    if bootstrap:
        indices = random_state.randint(0, n_population, n_samples)
    else:
        indices = sample_without_replacement(
            n_population, n_samples, random_state=random_state
        )

    return indices


def _generate_bagging_indices(
    random_state,
    bootstrap_features,
    bootstrap_samples,
    n_features,
    n_samples,
    max_features,
    max_samples,
):
    """Randomly draw feature and sample indices."""
    # Get valid random state
    random_state = check_random_state(random_state)

    # Draw indices
    feature_indices = _generate_indices(
        random_state, bootstrap_features, n_features, max_features
    )
    sample_indices = _generate_indices(
        random_state, bootstrap_samples, n_samples, max_samples
    )

    return feature_indices, sample_indices


def _parallel_build_estimators(
    n_estimators,
    ensemble,
    X,
    y,
    sample_weight,
    seeds,
    total_n_estimators,
    verbose, 
    check_input,
    ### KF:
    chol_eps=None, 
    do_param_boot=False,
    ###
):
    """Private function used to build a batch of estimators within a job."""
    # Retrieve settings
    n_samples, n_features = X.shape
    max_features = ensemble._max_features
    max_samples = ensemble._max_samples
    bootstrap = ensemble.bootstrap
    bootstrap_features = ensemble.bootstrap_features
    support_sample_weight = has_fit_parameter(ensemble.estimator_, "sample_weight")
    has_check_input = has_fit_parameter(ensemble.estimator_, "check_input")
    requires_feature_indexing = bootstrap_features or max_features != n_features

    if not support_sample_weight and sample_weight is not None:
        raise ValueError("The base estimator doesn't support sample weight")

    # Build estimators
    estimators = []
    estimators_features = []
    ### KF:
    eps_list = []
    ###

    for i in range(n_estimators):
        if verbose > 1:
            print(
                "Building estimator %d of %d for this parallel run (total %d)..."
                % (i + 1, n_estimators, total_n_estimators)
            )

        random_state = seeds[i]
        estimator = ensemble._make_estimator(append=False, random_state=random_state)

        if has_check_input:
            estimator_fit = partial(estimator.fit, check_input=check_input)
        else:
            estimator_fit = estimator.fit

        # Draw random feature, sample indices
        features, indices = _generate_bagging_indices(
            random_state,
            bootstrap_features,
            bootstrap,
            n_features,
            n_samples,
            max_features,
            max_samples,
        )

        # Draw samples, using sample weights, and then fit
        if support_sample_weight:
            if sample_weight is None:
                curr_sample_weight = np.ones((n_samples,))
            else:
                curr_sample_weight = sample_weight.copy()

            if bootstrap:
                ### KF:
                if do_param_boot:
                    if chol_eps is None:
                        eps = np.random.randn(n_samples)
                    else:
                        eps = np.random.randn(chol_eps.shape[0])
                        eps = chol_eps @ eps

                    w = y.flatten() + eps


                    eps_list.append(eps)
                else:
                    sample_counts = np.bincount(indices, minlength=n_samples)
                    curr_sample_weight *= sample_counts
                ###
            else:
                not_indices_mask = ~indices_to_mask(indices, n_samples)
                curr_sample_weight[not_indices_mask] = 0

            X_ = X[:, features] if requires_feature_indexing else X
            ### KF:
            estimator.fit(X_[:, features], w if do_param_boot else y, sample_weight=curr_sample_weight)
            ###

        else:
            X_ = X[indices][:, features] if requires_feature_indexing else X[indices]
            estimator_fit(X_, y[indices])

        estimators.append(estimator)
        estimators_features.append(features)

    return estimators, estimators_features, eps_list


class ParametricBaggingRegressor(LinearSelector, BaggingRegressor):

    def fit(
        self, 
        X, 
        y, 
        sample_weight=None, 
        chol_eps=None, 
        do_param_boot=True
    ):
        self.do_param_boot = do_param_boot
        self.X_tr_ = X
        _raise_for_unsupported_routing(self, "fit", sample_weight=sample_weight)
        # Convert data (X is required to be 2d and indexable)
        X, y = self._validate_data(
            X,
            y,
            accept_sparse=["csr", "csc"],
            dtype=None,
            force_all_finite=False,
            multi_output=True,
        )
        ### KF:
        return self._fit(X, y, max_samples=self.max_samples, sample_weight=sample_weight, chol_eps=chol_eps, do_param_boot=do_param_boot)
        ###
    
    def _fit(
        self,
        X,
        y,
        max_samples=None,
        max_depth=None,
        sample_weight=None,
        check_input=True,
        chol_eps=None, 
        do_param_boot=False
    ):
        """Build a Bagging ensemble of estimators from the training
           set (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.

        y : array-like of shape (n_samples,)
            The target values (class labels in classification, real numbers in
            regression).

        max_samples : int or float, default=None
            Argument to use instead of self.max_samples.

        max_depth : int, default=None
            Override value used when constructing base estimator. Only
            supported if the base estimator has a max_depth parameter.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted.
            Note that this is supported only if the base estimator supports
            sample weighting.

        check_input : bool, default=True
            Override value used when fitting base estimator. Only supported
            if the base estimator has a check_input parameter for fit function.

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
        # print("start bagging fit")
        random_state = check_random_state(self.random_state)

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X, dtype=None)

        # Remap output
        n_samples = X.shape[0]
        self._n_samples = n_samples
        y = self._validate_y(y)

        # Check parameters
        self._validate_estimator()

        if max_depth is not None:
            self.estimator_.max_depth = max_depth

        # Validate max_samples
        if max_samples is None:
            max_samples = self.max_samples
        elif not isinstance(max_samples, numbers.Integral):
            max_samples = int(max_samples * X.shape[0])

        if max_samples > X.shape[0]:
            raise ValueError("max_samples must be <= n_samples")

        # Store validated integer row sampling value
        self._max_samples = max_samples

        # Validate max_features
        if isinstance(self.max_features, numbers.Integral):
            max_features = self.max_features
        elif isinstance(self.max_features, float):
            max_features = int(self.max_features * self.n_features_in_)

        if max_features > self.n_features_in_:
            raise ValueError("max_features must be <= n_features")

        max_features = max(1, int(max_features))

        # Store validated integer feature sampling value
        self._max_features = max_features

        # Other checks
        if not self.bootstrap and self.oob_score:
            raise ValueError("Out of bag estimation only available if bootstrap=True")

        if self.warm_start and self.oob_score:
            raise ValueError("Out of bag estimate only available if warm_start=False")

        if hasattr(self, "oob_score_") and self.warm_start:
            del self.oob_score_

        if not self.warm_start or not hasattr(self, "estimators_"):
            # Free allocated memory, if any
            self.estimators_ = []
            self.estimators_features_ = []

        ### KF:
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
            return self

        # Parallel loop
        n_jobs, n_estimators, starts = _partition_estimators(
            n_more_estimators, self.n_jobs
        )
        total_n_estimators = sum(n_estimators)

        # Advance random state to state after training
        # the first n_estimators
        if self.warm_start and len(self.estimators_) > 0:
            random_state.randint(MAX_INT, size=len(self.estimators_))

        seeds = random_state.randint(MAX_INT, size=n_more_estimators)
        self._seeds = seeds

        all_results = Parallel(
            n_jobs=n_jobs, verbose=self.verbose, **self._parallel_args()
        )(
            delayed(_parallel_build_estimators)(
                n_estimators[i],
                self,
                X,
                y,
                sample_weight,
                seeds[starts[i] : starts[i + 1]],
                total_n_estimators,
                verbose=self.verbose,
                check_input=check_input,
                ### KF:
                chol_eps=chol_eps,
                do_param_boot=do_param_boot,
                ###
            )
            for i in range(n_jobs)
        )


        # Reduce
        self.estimators_ += list(
            itertools.chain.from_iterable(t[0] for t in all_results)
        )
        self.estimators_features_ += list(
            itertools.chain.from_iterable(t[1] for t in all_results)
        )
        ### KF:
        if do_param_boot:
            self.eps_ += list(
                itertools.chain.from_iterable(t[2] for t in all_results)
            )
        ###

        if self.oob_score:
            self._set_oob_score(X, y)

        return self

    def get_group_X(self, X, X_pred=None):
        check_is_fitted(self)

        return super().get_group_X(X)

    def get_linear_smoother(self, X, tr_idx, ts_idx, ret_full_P=False):
        assert(isinstance(self.base_estimator, LinearSelector))

        Ps = [est.get_linear_smoother(X, tr_idx, ts_idx, ret_full_P)[0] for est in self.estimators_]
        
        return Ps