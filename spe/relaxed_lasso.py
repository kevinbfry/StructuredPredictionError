import numpy as np
from sklearn.linear_model import LinearRegression, Lasso

from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.ensemble import BaggingRegressor

from .tree import LinearSelector

class BaggedRelaxedLasso(BaggingRegressor):
	def get_linear_smoother(self, X, X_pred=None):
		return [est.get_linear_smoother(X, X_pred) for est in self.estimators_]

	def get_group_X(self, X):
		check_is_fitted(self)

		n = X.shape[0]

		Gs = []
		for i in np.arange(self.n_estimators):
			est = self.estimators_[i]

			E = est.E_
			if E.shape[0] != 0:
				XE = X[:,E]
			else:
				XE = np.zeros((X.shape[0], 1))

			return Gs.append(XE)

		return Gs

			

class RelaxedLasso(LinearSelector, BaseEstimator):
	def __init__(self, 
			   lambd=1.0, 
			   fit_intercept=False, 
			   #  lasso_type='relaxed',
			   #  refit_type='full',
			   normalize='deprecated', 
			   precompute=False, 
			   copy_X=True, 
			   max_iter=1000, 
			   tol=0.0001, 
			   warm_start=False, 
			   positive=False, 
			   random_state=None, 
			   selection='cyclic'):
		(self.lambd, 
		 self.fit_intercept,
		 # self.lasso_type,
		 # self.refit_type,
		 self.normalize,
		 self.precompute,
		 self.copy_X,
		 self.max_iter,
		 self.tol,
		 self.warm_start,
		 self.positive,
		 self.random_state,
		 self.selection) = (lambd, 
							fit_intercept,
							# lasso_type,
							# refit_type,
							normalize,
							precompute,
							copy_X,
							max_iter,
							tol,
							warm_start,
							positive,
							random_state,
							selection)

		self.lassom = Lasso(alpha=lambd, 
							fit_intercept=fit_intercept,
							normalize=normalize,
							precompute=precompute,
							copy_X=copy_X,
							max_iter=max_iter,
							tol=tol,
							warm_start=warm_start,
							positive=positive,
							random_state=random_state,
							selection=selection)

		self.linm = LinearRegression(fit_intercept=fit_intercept,
									 copy_X=copy_X,
									 positive=positive)


	def get_group_X(self, X):
		check_is_fitted(self)

		E = self.E_
		if E.shape[0] != 0:
			XE = X[:,E]
		else:
			XE = np.zeros((X.shape[0], 1))

		return XE


	def get_linear_smoother(self, X, X_pred=None):
		XE = self.get_group_X(X)
		if not np.any(XE):
			print("zeros")
			return np.zeros((X_pred.shape[0], X.shape[0]))
		if X_pred is None:
			XE_pred = XE
		else:
			XE_pred = self.get_group_X(X_pred)
		# return XE_pred @ np.linalg.inv(XE.T @ XE) @ XE.T
		return XE_pred @ np.linalg.pinv(XE)


	def fit(self, X,
		  lasso_y,
		  lin_y=None,
		  sample_weight=None,
		  check_input=True):

		if lin_y is None:
			lin_y = lasso_y.copy()


		self.lassom.fit(X, 
						lasso_y,
						sample_weight=sample_weight,
						check_input=check_input)

		self.E_ = E = np.where(self.lassom.coef_ != 0)[0]

		self.fit_linear(X, 
						lin_y, 
						sample_weight=sample_weight)


		return self

	def fit_linear(self, 
				 X, 
				 y, 
				 sample_weight=None):
		XE = self.get_group_X(X)

		self.linm.fit(XE,
					  y,
					  sample_weight=sample_weight)


	def predict(self, X):
		XE = self.get_group_X(X)
		return self.linm.predict(XE)

