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
    """BSpline linear regression model.

    Generate feature matrix of B-splines and fit a linear regression to that 
    expanded feature set.

    Parameters
    ----------
    n_knots : int, optional
        Number of knots of the splines if `knots` equals one of
        {'uniform', 'quantile'}. Must be larger or equal 2. Ignored if ``knots``
        is array-like. Default is `5`.

    degree : int, optional
        The polynomial degree of the spline basis. Must be a non-negative
        integer. Default is ``3``.

    knots : {'uniform', 'quantile'} or array-like of shape \
        (n_knots, n_features), optional
        Set knot positions such that first knot <= features <= last knot.
        Default is `'uniform'`.

        - If `'uniform'`, ``n_knots`` number of knots are distributed uniformly
          from min to max values of the features.
        - If `'quantile'`, they are distributed uniformly along the quantiles of
          the features.
        - If an array-like is given, it directly specifies the sorted knot
          positions including the boundary knots. Note that, internally,
          ``degree`` number of knots are added before the first knot, the same
          after the last knot.

    extrapolation : {'error', 'constant', 'linear', 'continue', 'periodic'}, optional
        If 'error', values outside the min and max values of the training
        features raises a `ValueError`. If 'constant', the value of the
        splines at minimum and maximum value of the features is used as
        constant extrapolation. If 'linear', a linear extrapolation is used.
        If 'continue', the splines are extrapolated as is, i.e. option
        `extrapolate=True` in :class:`scipy.interpolate.BSpline`. If
        'periodic', periodic splines with a periodicity equal to the distance
        between the first and last knot are used. Periodic splines enforce
        equal function values and derivatives at the first and last knot.
        For example, this makes it possible to avoid introducing an arbitrary
        jump between Dec 31st and Jan 1st in spline features derived from a
        naturally periodic "day-of-year" input feature. In this case it is
        recommended to manually set the knot values to control the period.
        Default is `'constant'`.

    include_bias : bool, optional
        If False, then the last spline element inside the data range
        of a feature is dropped. As B-splines sum to one over the spline basis
        functions for each data point, they implicitly include a bias term,
        i.e. a column of ones. It acts as an intercept term in a linear models.
        Default is ``True``.

    order : {'C', 'F'}, optional
        Order of output array in the dense case. `'F'` order is faster to compute, but
        may slow down subsequent estimators. Default is `'C'`.

    sparse_output : bool, optional
        Will return sparse CSR matrix if set True else will return an array. This
        option is only available with ``scipy>=1.8``. Default is ``False``.
    """
    def __init__(
        self,
        n_knots=5, 
        degree=3,
        knots='uniform', 
        extrapolation='linear', 
        include_bias=True, 
        order='C',
        sparse_output=False,
    ):
        self.n_knots=n_knots
        self.degree=degree
        self.knots=knots
        self.extrapolation=extrapolation
        self.include_bias=include_bias
        self.order=order
        self.sparse_output=sparse_output

    def get_group_X(self, X, X_pred=None):
        return super().get_group_X(X, X_pred)
    
    def get_linear_smoother(
        self, 
        X, 
        tr_idx,
        ts_idx, 
        ret_full_P=False
    ):
        X_tr = self.spline_transformer_.transform(X[tr_idx,:])
        X_ts = self.spline_transformer_.transform(X[ts_idx,:])
        if ret_full_P:
            n = X.shape[0]
            full_X_tr = np.zeros((n,X_tr.shape[1]))
            full_X_tr[tr_idx,:] = X_tr
            return [X_ts @ np.linalg.pinv(full_X_tr)]
        return [X_ts @ np.linalg.pinv(X_tr)]
    
    def fit(self, X, y):
        self.spline_transformer_ = SplineTransformer(
            n_knots=self.n_knots, 
            degree=self.degree, 
            knots=self.knots, 
            extrapolation=self.extrapolation, 
            include_bias=self.include_bias, 
            order=self.order,
        )
        self.model_ = LinReg()

        X_spline = self.spline_transformer_.fit_transform(X)
        self.model_.fit(X_spline, y)
        self.fitted_flag_ = True

    def predict(self, X):
        check_is_fitted(self)
        X_spline = self.spline_transformer_.transform(X)
        return self.model_.predict(X_spline)
