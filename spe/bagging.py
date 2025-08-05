from functools import partial
import itertools
import numbers
from warnings import warn

import numpy as np

from scipy.sparse import issparse

from sklearn.ensemble import BaggingRegressor
from sklearn.utils import check_random_state
# from sklearn.utils.validation import validate_data
from sklearn.utils._mask import indices_to_mask
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

from .smoothers import AdaptiveLinearSmoother

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


class ParametricBaggingRegressor(AdaptiveLinearSmoother, BaggingRegressor):
    """A Bagging regressor.

    A Bagging regressor is an ensemble meta-estimator that fits base
    regressors each on random subsets of the original dataset and then
    aggregate their individual predictions (either by voting or by averaging)
    to form a final prediction. Such a meta-estimator can typically be used as
    a way to reduce the variance of a black-box estimator (e.g., a decision
    tree), by introducing randomization into its construction procedure and
    then making an ensemble out of it.

    This algorithm encompasses several works from the literature. When random
    subsets of the dataset are drawn as random subsets of the samples, then
    this algorithm is known as Pasting [1]_. If samples are drawn with
    replacement, then the method is known as Bagging [2]_. When random subsets
    of the dataset are drawn as random subsets of the features, then the method
    is known as Random Subspaces [3]_. Finally, when base estimators are built
    on subsets of both samples and features, then the method is known as
    Random Patches [4]_.

    Additionally, can fit with Gaussian parametric bootstraps, and is a subclass of
    :class:`~spe.tree.AdaptiveLinearSmoother`.

    Documentation is heavily lifted from :class:`~sklearn.ensemble.BaggingRegressor` class, which 
    this class inherits from.

    Parameters
    ----------
    estimator : object, default=None
        The base estimator to fit on random subsets of the dataset.
        If None, then the base estimator is a
        :class:`~sklearn.tree.DecisionTreeRegressor`.

    n_estimators : int, default=10
        The number of base estimators in the ensemble.

    max_samples : int or float, default=1.0
        The number of samples to draw from X to train each base estimator (with
        replacement by default, see `bootstrap` for more details).

        - If int, then draw `max_samples` samples.
        - If float, then draw `max_samples * X.shape[0]` samples.

    max_features : int or float, default=1.0
        The number of features to draw from X to train each base estimator (
        without replacement by default, see `bootstrap_features` for more
        details).

        - If int, then draw `max_features` features.
        - If float, then draw `max(1, int(max_features * n_features_in_))` features.

    bootstrap : bool, default=True
        Whether samples are drawn with replacement. If False, sampling
        without replacement is performed.

    bootstrap_features : bool, default=False
        Whether features are drawn with replacement.

    oob_score : bool, default=False
        Whether to use out-of-bag samples to estimate
        the generalization error. Only available if bootstrap=True.

    warm_start : bool, default=False
        When set to True, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit
        a whole new ensemble.

    n_jobs : int, default=None
        The number of jobs to run in parallel for both :meth:`fit` and
        :meth:`predict`. ``None`` means 1 unless in a
        :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors.

    random_state : int, RandomState instance or None, default=None
        Controls the random resampling of the original dataset
        (sample wise and feature wise).
        If the base estimator accepts a `random_state` attribute, a different
        seed is generated for each instance in the ensemble.
        Pass an int for reproducible output across multiple function calls.

    verbose : int, default=0
        Controls the verbosity when fitting and predicting.

    Attributes
    ----------
    estimator_ : estimator
        The base estimator from which the ensemble is grown.


    n_features_in_ : int
        Number of features seen during ``fit``.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during ``fit``. Defined only when `X`
        has feature names that are all strings.

    estimators_ : list of estimators
        The collection of fitted sub-estimators.

    estimators_samples_ : list of arrays
        The subset of drawn samples (i.e., the in-bag samples) for each base
        estimator. Each subset is defined by an array of the indices selected.

    estimators_features_ : list of arrays
        The subset of drawn features for each base estimator.

    oob_score_ : float
        Score of the training dataset obtained using an out-of-bag estimate.
        This attribute exists only when ``oob_score`` is True.

    oob_prediction_ : ndarray of shape (n_samples,)
        Prediction computed with out-of-bag estimate on the training
        set. If n_estimators is small it might be possible that a data point
        was never left out during the bootstrap. In this case,
        `oob_prediction_` might contain NaN. This attribute exists only
        when ``oob_score`` is True.

    See Also
    --------
    BaggingClassifier : A Bagging classifier.

    References
    ----------

    .. [1] L. Breiman, "Pasting small votes for classification in large
           databases and on-line", Machine Learning, 36(1), 85-103, 1999.

    .. [2] L. Breiman, "Bagging predictors", Machine Learning, 24(2), 123-140,
           1996.

    .. [3] T. Ho, "The random subspace method for constructing decision
           forests", Pattern Analysis and Machine Intelligence, 20(8), 832-844,
           1998.

    .. [4] G. Louppe and P. Geurts, "Ensembles on Random Patches", Machine
           Learning and Knowledge Discovery in Databases, 346-361, 2012.

    Examples
    --------
    >>> from sklearn.svm import SVR
    >>> from spe.bagging import BaggingRegressor
    >>> from sklearn.datasets import make_regression
    >>> X, y = make_regression(n_samples=100, n_features=4,
    ...                        n_informative=2, n_targets=1,
    ...                        random_state=0, shuffle=False)
    >>> regr = ParametricBaggingRegressor(estimator=SVR(),
    ...                         n_estimators=10, random_state=0).fit(X, y)
    >>> regr.predict([[0, 0, 0, 0]])
    array([-2.8720...])
    """
    def fit(
        self,
        X,
        y,
        max_samples=None,
        max_depth=None,
        sample_weight=None,
        check_input=True,
        chol_eps=None, 
        do_param_boot=False,
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
        self.do_param_boot = do_param_boot
        self.X_tr_ = X
        _raise_for_unsupported_routing(self, "fit", sample_weight=sample_weight)
        # Convert data (X is required to be 2d and indexable)
        X, y = self._validate_data(
        # X, y = validate_data(
            X,
            y,
            accept_sparse=["csr", "csc"],
            dtype=None,
            force_all_finite=False,
            # ensure_all_finite=False,
            multi_output=True,
        )
        ### KF:
        return self._fit(
            X,
            y,
            max_samples=max_samples if max_samples is not None else self.max_samples,
            max_depth=max_depth,
            sample_weight=sample_weight,
            check_input=check_input,
            chol_eps=chol_eps, 
            do_param_boot=do_param_boot,
        )
        ###
    
    def _fit(
        self,
        X,
        y,
        max_samples,
        max_depth,
        sample_weight,
        check_input,
        chol_eps, 
        do_param_boot,
    ):
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

    def get_linear_smoother(self, X, tr_idx, ts_idx, ret_full_P=False):
        assert(isinstance(self.base_estimator, AdaptiveLinearSmoother))

        Ps = [est.get_linear_smoother(X, tr_idx, ts_idx, ret_full_P)[0] for est in self.estimators_]
        
        return Ps