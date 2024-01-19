import numpy as np
from sklearn.linear_model import LinearRegression, Lasso

from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn.ensemble import BaggingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .tree import LinearSelector


class RelaxedLasso(LinearSelector, BaseEstimator):
    def __init__(
        self,
        lambd=1.0,
        fit_intercept=False,
        precompute=False,
        copy_X=True,
        max_iter=1000,
        tol=0.0001,
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