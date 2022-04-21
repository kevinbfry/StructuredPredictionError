from itertools import product

import numpy as np

from scipy.linalg import block_diag
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import cross_validate, GroupKFold, KFold
from sklearn.cluster import KMeans
from sklearn.base import clone

from .relaxed_lasso import RelaxedLasso
from .estimators import kfoldcv, kmeanscv, test_set_estimator, cb, cb_isotropic, blur_linear, blur_lasso
from .tree import Tree, BlurTreeIID

class ErrorComparer(object):
	def compareIID(self, 
					 niter=100,
					 n=100,
					 p=200,
					 s=5,
					 snr=0.4, 
					 X=None,
					 beta=None,
					 model=Lasso(),
					 alpha=0.05,
					 est_risk=True):


		self.test_err = np.zeros(niter)
		self.test_err_alpha = np.zeros(niter)
		self.cb_err = np.zeros(niter)
		self.cbiso_err = np.zeros(niter)

		test_est = test_set_estimator
		cb_est = cb
		cbiso_est = cb_isotropic()

		gen_beta = X is None or beta is None

		if not gen_beta:
			mu = X @ beta
			sigma = np.sqrt(np.var(mu)/snr)
			n, p = X.shape

		for i in np.arange(niter):
		
			if gen_beta:
				X = np.random.randn(n,p)
				beta = np.zeros(p)
				idx = np.random.choice(p,size=s)
				beta[idx] = np.random.uniform(-1,1,size=s)

				mu = X @ beta
				sigma = np.sqrt(np.var(mu)/snr)

			y = mu + sigma * np.random.randn(n)
			y_test = mu + sigma * np.random.randn(n)
			y_alpha = mu + sigma * np.sqrt(1 + alpha) * np.random.randn(n)
			y_test_alpha = mu + sigma * np.sqrt(1 + alpha) * np.random.randn(n)

			self.test_err[i] = test_est(model=model,
										X=X, 
										y=y, 
										y_test=y_test, 
										Chol_t=np.eye(n)*sigma, 
										est_risk=est_risk)[0]
			self.test_err_alpha[i] = test_est(model=model,
											  X=X, 
											  y=y_alpha, 
											  y_test=y_test_alpha, 
											  Chol_t=np.eye(n)*np.sqrt(1+alpha)*sigma, 
											  est_risk=est_risk)[0]
			self.cb_err[i] = cb_est(X=X, 
									y=y, 
									Chol_t=np.eye(n)*sigma, 
									Chol_eps=np.eye(n)*np.sqrt(alpha)*sigma,
									model=model,
									est_risk=est_risk)[0]
			self.cbiso_err[i] = cbiso_est(X,
										  y,
										  sigma=sigma,
										  alpha=alpha,
										  model=model,
										  est_risk=est_risk)[0]

		return (self.test_err,
				self.test_err_alpha,
				self.cb_err,
				self.cbiso_err)

	def compareTreeIID(self, 
						 niter=100,
						 n=100,
						 p=200,
						 s=5,
						 snr=0.4, 
						 X=None,
						 beta=None,
						 model=Tree(),
						 rand_type='full',
						 alpha=0.05,
						 est_risk=True):


		self.test_err = np.zeros(niter)
		self.blur_err = np.zeros(niter)

		test_est = test_set_estimator
		blur_est = BlurTreeIID()

		gen_beta = X is None or beta is None

		if not gen_beta:
			mu = X @ beta
			sigma = np.sqrt(np.var(mu)/snr)
			n, p = X.shape

		for i in np.arange(niter):
		
			if gen_beta:
				X = np.random.randn(n,p)
				beta = np.zeros(p)
				idx = np.random.choice(p,size=s)
				beta[idx] = np.random.uniform(-1,1,size=s)

				mu = X @ beta
				sigma = np.sqrt(np.var(mu)/snr)

			y = mu + sigma * np.random.randn(n)
			y_test = mu + sigma * np.random.randn(n)

			(self.blur_err[i], fitted_model) = blur_est._estimate(X, 
																y, 
																Chol_t=np.eye(n)*sigma, 
																Chol_eps=np.eye(n)*np.sqrt(alpha)*sigma,
																model=model,
																rand_type=rand_type,
																est_risk=est_risk)

			Z = fitted_model.get_membership_matrix(X)
			y_fit = y if rand_type == 'full' else blur_est.w
			self.test_err[i] = test_est(model=LinearRegression(),
										X=Z,
										y=y_fit,
										y_test=y_test, 
										Chol_t=np.eye(n)*sigma, 
										est_risk=est_risk)[0]

		return (self.test_err,
				self.blur_err)

	def compareBlurLinearIID(self, 
							 niter=100,
							 n=100,
							 p=20,
							 s=20,
							 snr=0.4, 
							 X=None,
							 beta=None,
							 model=LinearRegression(),
							 alpha=0.05,
							 est_risk=True):


		self.test_err = np.zeros(niter)
		self.test_err_alpha = np.zeros(niter)
		self.cb_err = np.zeros(niter)
		self.blur_err = np.zeros(niter)

		test_est = test_set_estimator
		cb_est = cb
		blur_est = blur_linear

		gen_beta = X is None or beta is None

		if not gen_beta:
			mu = X @ beta
			sigma = np.sqrt(np.var(mu)/snr)
			n, p = X.shape

		for i in np.arange(niter):
		
			if gen_beta:
				X = np.random.randn(n,p)
				beta = np.zeros(p)
				idx = np.random.choice(p,size=s)
				beta[idx] = np.random.uniform(-1,1,size=s)

				mu = X @ beta
				sigma = np.sqrt(np.var(mu)/snr)

			y = mu + sigma * np.random.randn(n)
			y_test = mu + sigma * np.random.randn(n)
			y_alpha = mu + sigma * np.sqrt(1 + alpha) * np.random.randn(n)
			y_test_alpha = mu + sigma * np.sqrt(1 + alpha) * np.random.randn(n)

			self.test_err[i] = test_est(model=model,
										X=X, 
										y=y, 
										y_test=y_test, 
										Chol_t=np.eye(n)*sigma, 
										est_risk=est_risk)[0]
			self.test_err_alpha[i] = test_est(model=model,
											  X=X, 
											  y=y_alpha, 
											  y_test=y_test_alpha, 
											  Chol_t=np.eye(n)*np.sqrt(1+alpha)*sigma, 
											  est_risk=est_risk)[0]
			self.cb_err[i] = cb_est(X=X, 
									y=y, 
									Chol_t=np.eye(n)*sigma, 
									Chol_eps=np.eye(n)*np.sqrt(alpha)*sigma,
									model=model,
									est_risk=est_risk)[0]
			self.blur_err[i] = blur_est(X, 
										y, 
										Chol_t=np.eye(n)*sigma, 
										Chol_eps=np.eye(n)*np.sqrt(alpha)*sigma,
										model=model,
										est_risk=est_risk)[0]

		return (self.test_err,
				self.test_err_alpha,
				self.cb_err,
				self.blur_err)

	def compareBlurLassoIID(self, 
							 niter=100,
							 n=100,
							 p=20,
							 s=20,
							 snr=0.4, 
							 X=None,
							 beta=None,
							 model=RelaxedLasso(),
							 rand_type='full',
							 alpha=0.05,
							 est_risk=True):


		self.test_err = np.zeros(niter)
		self.cb_err = np.zeros(niter)
		self.blur_err = np.zeros(niter)

		test_est = test_set_estimator
		blur_est = blur_lasso

		gen_beta = X is None or beta is None

		if not gen_beta:
			mu = X @ beta
			sigma = np.sqrt(np.var(mu)/snr)
			n, p = X.shape

		for i in np.arange(niter):
		
			if gen_beta:
				X = np.random.randn(n,p)
				beta = np.zeros(p)
				idx = np.random.choice(p,size=s)
				beta[idx] = np.random.uniform(-1,1,size=s)

				mu = X @ beta
				sigma = np.sqrt(np.var(mu)/snr)

			y = mu + sigma * np.random.randn(n)
			y_test = mu + sigma * np.random.randn(n)
			(self.blur_err[i],
			 fitted_model) = blur_est(X, 
									y, 
									Chol_t=np.eye(n)*sigma, 
									Chol_eps=np.eye(n)*np.sqrt(alpha)*sigma,
									model=model,
									est_risk=est_risk)

			XE = X[:, fitted_model.E_] if fitted_model.E_.shape[0] != 0 else np.zeros((X.shape[0],1))
			self.test_err[i] = test_est(model=LinearRegression(),
										X=XE, 
										y=y, 
										y_test=y_test, 
										Chol_t=np.eye(n)*sigma, 
										est_risk=est_risk)[0]

		return (self.test_err,
				self.blur_err)

	








