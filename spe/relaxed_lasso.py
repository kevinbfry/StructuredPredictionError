import numpy as np
from sklearn.linear_model import LinearRegression, Lasso

from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn.ensemble import BaggingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .tree import LinearSelector


class RelaxedLasso(LinearSelector, BaseEstimator):
    """Relaxed lasso linear regression model.

    Fits the usual lasso, then refits an unpenalized linear regression on features
    selected by the lasso.

    Documentation is heavily lifted from sklearn Lasso and LinearRegression classes,
    both of which are utilized by this class.

    Parameters
    ----------
    lambd : float, optional
        Constant that multiplies the L1 term, controlling regularization strength. 
        ``lambd`` must be a non-negative ``float`` i.e. in ``[0, inf)``.

    fit_intercept : bool, optional  
        Whether to calculate the intercept for this model. If set to ``False``, no 
        intercept will be used in calculations (i.e. data is expected to be centered). 
        Default is ``True``.

    precompute : bool or array-like of shape (n_features, n_features), optional
        Whether to use a precomputed Gram matrix to speed up calculations. The 
        Gram matrix can also be passed as argument. For sparse input this option 
        is always ``False`` to preserve sparsity. Default is ``False``.

    copy_X : bool, optional
        If True, ``X`` will be copied; else, it may be overwritten. Default is ``True``.


    max_iter : int, optional
        The maximum number of iterations. Default is ``1000``.

    tol : float, optional
        The tolerance for the optimization: if the updates are smaller than ``tol``, 
        the optimization code checks the dual gap for optimality and continues until 
        it is smaller than ``tol``. Default is ``1e-4``.

    warm_start : bool, optional
        When set to ``True``, reuse the solution of the previous call to fit as 
        initialization, otherwise, just erase the previous solution. Default is ``False``.

    positive : bool, optional
        When set to ``True``, forces the coefficients to be positive. Default is ``False``.

    random_state : int, optional
        The seed of the pseudo random number generator that selects a random feature 
        to update. Used when ``selection`` is ``random``. Pass an ``int`` for reproducible 
        output across multiple function calls. Default is ``None``.

    selection : {'cyclic', 'random'}, optional
        If set to ``random``, a random coefficient is updated every iteration rather 
        than looping over features sequentially by default. This (setting to 
        ``random``) often leads to significantly faster convergence especially 
        when tol is higher than 1e-4. Default is `'cyclic'`.
        
    
    """
    def __init__(
        self,
        lambd=1.0,
        fit_intercept=False,
        precompute=False,
        copy_X=True,
        max_iter=1000,
        tol=1e-4,#0.0001,
        warm_start=False,
        positive=False,
        random_state=None,
        selection="cyclic",
    ):
        (
            self.lambd,
            self.fit_intercept,
            self.precompute,
            self.copy_X,
            self.max_iter,
            self.tol,
            self.warm_start,
            self.positive,
            self.random_state,
            self.selection,
        ) = (
            lambd,
            fit_intercept,
            precompute,
            copy_X,
            max_iter,
            tol,
            warm_start,
            positive,
            random_state,
            selection,
        )

    def get_group_X(self, X):
        check_is_fitted(self)

        E = self.E_
        if E.shape[0] != 0:
            XE = X[:, E]
        else:
            XE = np.zeros((X.shape[0], 1))

        return XE

    def get_linear_smoother(self, X, tr_idx, ts_idx, ret_full_P=False):
        X_tr = X[tr_idx,:]
        X_ts = X[ts_idx,:]
        XE_tr = self.get_group_X(X_tr)
        if not np.any(XE_tr):
            # print("zeros")
            if ret_full_P:
                return [np.zeros((X_ts.shape[0], X.shape[0]))]
            return [np.zeros((X_ts.shape[0], X_tr.shape[0]))]
        
        XE_ts = self.get_group_X(X_ts)
        if ret_full_P:
            n = X.shape[0]
            full_XE_tr = np.zeros((n,XE_tr.shape[1]))
            full_XE_tr[tr_idx,:] = XE_tr
            return [XE_ts @ np.linalg.pinv(full_XE_tr)]
        return [XE_ts @ np.linalg.pinv(XE_tr)]

    def fit(self, X, lasso_y, lin_y=None, sample_weight=None, check_input=True):
        self.lassom = Pipeline([
            # ('scaler', StandardScaler()),
            ('model', Lasso(
                alpha=self.lambd,
                fit_intercept=self.fit_intercept,
                precompute=self.precompute,
                copy_X=self.copy_X,
                max_iter=self.max_iter,
                tol=self.tol,
                warm_start=self.warm_start,
                positive=self.positive,
                random_state=self.random_state,
                selection=self.selection,
            ))
        ])

        self.linm = LinearRegression(
            fit_intercept=self.fit_intercept, copy_X=self.copy_X, positive=self.positive
        )

        if lin_y is None:
            self.lin_y = lin_y = lasso_y.copy()

        self.lassom.fit(
            X, 
            lasso_y, 
            model__sample_weight=sample_weight, 
            model__check_input=check_input,
        )

        self.E_ = E = np.where(self.lassom.named_steps['model'].coef_ != 0)[0]
        self.fit_linear(X, lin_y, sample_weight=sample_weight)

        return self

    def fit_linear(self, X, y, sample_weight=None):
        XE = self.get_group_X(X)

        self.linm.fit(XE, y, sample_weight=sample_weight)

    def predict(
        self,
        X,
        tr_idx=None,
        ts_idx=None,
        y_refit=None,
    ):
        check_is_fitted(self)
        if tr_idx is None and ts_idx is None and y_refit is None:
            XE = self.get_group_X(X)
            return self.linm.predict(XE)
        return super().predict(X, tr_idx, ts_idx, y_refit)