import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.validation import check_is_fitted


class Tree(DecisionTreeRegressor):
	def get_membership_matrix(self, X):
		check_is_fitted(self)

		leaf_nodes = self.apply(X)
		_, indices = np.unique(leaf_nodes, return_inverse=True)

		n_leaves = self.get_n_leaves()
		n = X.shape[0]

		Z = np.zeros((n, n_leaves))
		Z[np.arange(n),indices] = 1

		return Z

class BlurTreeIID(object):
	
	def _estimate(self,
					X, 
					y, 
					Chol_t=None, 
					Chol_eps=None,
					# Theta=None,
					model=Tree(),
					rand_type='full',
					est_risk=True):

		model = clone(model)

		X = X
		y = y
		(n, p) = X.shape

		full_rand = rand_type == 'full'

		if Chol_eps is None:
			Chol_eps = np.eye(n)
			Sigma_eps = Chol_eps
		else:
			Sigma_eps = Chol_eps @ Chol_eps.T
		
		Prec_eps = np.linalg.inv(Sigma_eps)

		if Chol_t is None:
			Chol_t = np.eye(n)
			Sigma_t = np.eye(n)
		else:
			Sigma_t = Chol_t @ Chol_t.T

		proj_t_eps = Sigma_t @ Prec_eps

		# if Theta is None:
		#	 Theta = np.eye(n)
		Sigma_t_Theta = Sigma_t# @ Theta

		Aperpinv = np.eye(proj_t_eps.shape[0]) + proj_t_eps
		Aperp = np.linalg.inv(Aperpinv)

		eps = Chol_eps @ np.random.randn(n)
		w = y + eps
		regress_t_eps = proj_t_eps @ eps
		wp = y - regress_t_eps

		model.fit(X, w)

		Z = model.get_membership_matrix(X)
		P = Z @ np.linalg.inv(Z.T @ Z) @ Z.T
		yhat = P @ y if full_rand else P @ w
		PAperp = P @ Aperp

		boot_est = np.sum((wp - yhat)**2)

		t_epsinv_t = proj_t_eps @ Sigma_t
		expectation_correction = - np.diag(t_epsinv_t).sum()
		if full_rand:
			expectation_correction += 2*np.diag((Sigma_t + t_epsinv_t) @ PAperp).sum()
		
		return (boot_est + expectation_correction
				- np.diag(Sigma_t_Theta).sum()*est_risk) / n, model, w









