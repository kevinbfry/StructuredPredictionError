import numpy as np

from sklearn.linear_model import LinearRegression as LinReg
from sklearn.preprocessing import SplineTransformer
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from .tree import LinearSelector


class LinearRegression(LinearSelector, LinReg):
    def get_group_X(self, X, X_pred=None):
        return super().get_group_X(X, X_pred)
    
    def get_linear_smoother(self, X, tr_idx, ts_idx, ret_full_P=False):
        if self.fit_intercept:
            X = np.hstack([np.ones((X.shape[0],1)), X])
        X_tr = X[tr_idx,:]
        X_ts = X[ts_idx,:]
        if ret_full_P:
            n = X.shape[0]
            full_X_tr = np.zeros((n,X_tr.shape[1]))
            full_X_tr[tr_idx,:] = X_tr
            return [X_ts @ np.linalg.pinv(full_X_tr)]
        return [X_ts @ np.linalg.pinv(X_tr)]


class BSplineRegressor(LinearSelector, BaseEstimator):
    # def __init__(
    #     self,
    #     n_knots=5, 
    #     degree=3,
    #     knots='uniform', 
    #     extrapolation='linear', 
    #     include_bias=True, 
    #     order='C',
    # ):
    #     self.n_knots=n_knots, 
    #     self.degree=degree, 
    #     self.knots=knots, 
    #     self.extrapolation=extrapolation, 
    #     self.include_bias=include_bias, 
    #     self.order=order

    def get_group_X(self, X, X_pred=None):
        return super().get_group_X(X, X_pred)
    
    def get_linear_smoother(
        self, 
        X, 
        tr_idx,
        ts_idx, 
        ret_full_P=False
    ):
        # X_spline = self.spline_transformer.transform(X)
        # X_tr = X[tr_idx,:]
        # X_ts = X[ts_idx,:]
        X_tr = self.spline_transformer.transform(X[tr_idx,:])
        X_ts = self.spline_transformer.transform(X[ts_idx,:])
        if ret_full_P:
            n = X.shape[0]
            full_X_tr = np.zeros((n,X_tr.shape[1]))
            full_X_tr[tr_idx,:] = X_tr
            return [X_ts @ np.linalg.pinv(full_X_tr)]
        return [X_ts @ np.linalg.pinv(X_tr)]
    
    def fit(self, X, y):
        self.spline_transformer = SplineTransformer(
            # n_knots=self.n_knots, 
            # degree=self.degree, 
            # knots=self.knots, 
            # extrapolation=self.extrapolation, 
            # include_bias=self.include_bias, 
            # order=self.order,
            n_knots=5, 
            degree=3,
            knots='uniform', 
            extrapolation='linear', 
            include_bias=True, 
            order='C',
        )
        self.model = LinReg()

        X_spline = self.spline_transformer.fit_transform(X)
        self.model.fit(X_spline, y)
        self.fitted_flag_ = True

    def predict(self, X):
        check_is_fitted(self)
        X_spline = self.spline_transformer.transform(X)
        return self.model.predict(X_spline)
