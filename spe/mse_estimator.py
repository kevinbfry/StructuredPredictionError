from itertools import product

import numpy as np

from scipy.linalg import block_diag, sqrtm
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import cross_validate, GroupKFold, KFold
from sklearn.cluster import KMeans
from sklearn.base import clone
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestRegressor

from .relaxed_lasso import RelaxedLasso
from .estimators import kfoldcv, kmeanscv, test_set_estimator, \
						cb, cb_isotropic, \
						blur_linear, blur_linear_selector, blur_forest, \
						cp_linear_train_test, test_est_split, \
						cp_relaxed_lasso_train_test, cp_random_forest_train_test
from .tree import Tree
from .forest import BlurredForest


## for 5/11 slides
def gen_rbf_X(c_x, c_y, p, nspikes=None):
	locs = np.stack([c_x, c_y]).T

	n = len(locs)

	rbf_kernel = RBF(5)
	cov = rbf_kernel(locs)

	if nspikes is None:
		nspikes = int(2*np.log2(n))

	mu = np.zeros((n, p))
	for i in np.arange(p):
		spikes = np.random.uniform(2,5, size=nspikes) * np.random.choice([-1,1],size=nspikes,replace=True)
		spike_idx = np.random.choice(n, size=nspikes)

		W = cov[:, spike_idx]
		W = W / W.sum(axis=1)[:,None]
		mu[:,i] = W @ spikes

	return mu


def create_clus_split(nx, ny):
	xv, yv = np.meshgrid(np.arange(nx), np.arange(ny))
	pts = np.stack([xv.ravel(), yv.ravel()]).T
	n = nx*ny
	rn = int(np.log2(n))
	ctr = np.random.choice(pts.shape[0], size=rn, replace=True)
	ctr = pts[ctr]
	tr_idx = np.vstack([[pt + np.array((1.25*np.random.randn(2)).astype(int)) for _ in np.arange(int(n/rn))] for pt in ctr])
	tr_idx = np.maximum(0, tr_idx)
	tr_idx[:,0] = cx = np.minimum(nx-1, tr_idx[:,0])
	tr_idx[:,1] = cy = np.minimum(ny-1, tr_idx[:,1])
	tr_idx = np.unique(np.ravel_multi_index(tr_idx.T, (nx,ny)))

	tr_bool = np.zeros(n).astype(bool)
	tr_bool[tr_idx] = True

	return tr_bool


class ErrorComparer(object):
	def compareLinearTrTs(
		self, 
		niter=100,
		n=200,
		p=30,
		s=5,
		snr=0.4, 
		X=None,
		beta=None,
		coord=None,
		Chol_t=None,
		tr_idx=None,
		k=10,
		):

		model = LinearRegression(fit_intercept=False)

		self.test_err = np.zeros(niter)
		self.kfcv_err = np.zeros(niter)
		self.spcv_err = np.zeros(niter)
		self.lin_err = np.zeros(niter)

		test_est = test_est_split
		kfcv_est = kfoldcv
		spcv_est = kmeanscv
		lin_est = cp_linear_train_test

		# gen_beta = X is None or beta is None

		# if not gen_beta:
		mu = X @ beta
		sigma = np.sqrt(np.var(mu)/snr)
		n, p = X.shape

		for i in np.arange(niter):
		
			# if gen_beta:
			# 	X = np.random.randn(n,p)
			# 	beta = np.zeros(p)
			# 	idx = np.random.choice(p,size=s)
			# 	beta[idx] = np.random.uniform(-1,1,size=s)

			# 	mu = X @ beta
			# 	sigma = np.sqrt(np.var(mu)/snr)

			tr_idx = create_clus_split(int(np.sqrt(n)), int(np.sqrt(n)))
			if i == 0:
				print(tr_idx.mean())
			# if tr_idx is None:
				# tr_samples = np.random.choice(n, size=int(.8*n), replace=False)
				# tr_idx = np.zeros(n)
				# tr_idx[tr_samples] = 1
				# ts_idx = (1 - tr_idx).astype(bool)

			eps = sigma * np.random.randn(n)
			eps2 = sigma * np.random.randn(n)
			if Chol_t is not None:
				eps = Chol_t @ eps
				eps2 = Chol_t @ eps2
			y = mu + eps
			y2 = mu + eps2

			X_tr = X[tr_idx,:]
			# X_ts = X[ts_idx,:]
			y_tr = y[tr_idx]
			# y_ts = y[ts_idx]
			coord_tr = coord[tr_idx,:]
			# return X_tr, y_tr, coord_tr

			self.test_err[i] = test_est(model=model,
										X=X, 
										y=y, 
										y2=y2,
										tr_idx=tr_idx)

			self.kfcv_err[i] = kfcv_est(model=model,
										X=X_tr, 
										y=y_tr,
										k=k)[0]

			self.spcv_err[i] = spcv_est(model=model,
										X=X_tr, 
										y=y_tr,
										coord=coord_tr,
										k=k)[0]

			self.lin_err[i] = lin_est(X=X, 
									  y=y, 
									  tr_idx=tr_idx,
									  Chol_t=Chol_t*sigma)

		return self.test_err, self.kfcv_err, self.spcv_err, self.lin_err


	def compareRelaxedLassoTrTs(
		self, 
		niter=100,
		n=200,
		p=30,
		s=5,
		snr=0.4, 
		X=None,
		beta=None,
		coord=None,
		Chol_t=None,
		alpha=1.,
		tr_idx=None,
		k=10,
		):

		model = RelaxedLasso(fit_intercept=False)

		self.test_err = np.zeros(niter)
		self.kfcv_err = np.zeros(niter)
		self.spcv_err = np.zeros(niter)
		self.rela_err = np.zeros(niter)

		test_est = test_est_split
		kfcv_est = kfoldcv
		spcv_est = kmeanscv
		rela_est = cp_relaxed_lasso_train_test

		# gen_beta = X is None or beta is None

		# if not gen_beta:
		mu = X @ beta
		sigma = np.sqrt(np.var(mu)/snr)
		n, p = X.shape

		for i in np.arange(niter):
		
			# if gen_beta:
			# 	X = np.random.randn(n,p)
			# 	beta = np.zeros(p)
			# 	idx = np.random.choice(p,size=s)
			# 	beta[idx] = np.random.uniform(-1,1,size=s)

			# 	mu = X @ beta
			# 	sigma = np.sqrt(np.var(mu)/snr)

			tr_idx = create_clus_split(int(np.sqrt(n)), int(np.sqrt(n)))
			if i == 0:
				print(tr_idx.mean())
			# if tr_idx is None:
				# tr_samples = np.random.choice(n, size=int(.8*n), replace=False)
				# tr_idx = np.zeros(n)
				# tr_idx[tr_samples] = 1
				# ts_idx = (1 - tr_idx).astype(bool)

			eps = sigma * np.random.randn(n)
			eps2 = sigma * np.random.randn(n)
			if Chol_t is not None:
				eps = Chol_t @ eps
				eps2 = Chol_t @ eps2
			y = mu + eps
			y2 = mu + eps2

			X_tr = X[tr_idx,:]
			# X_ts = X[ts_idx,:]
			y_tr = y[tr_idx]
			# y_ts = y[ts_idx]
			coord_tr = coord[tr_idx,:]
			# return X_tr, y_tr, coord_tr

			self.test_err[i] = test_est(model=model,
										X=X, 
										y=y, 
										y2=y2,
										tr_idx=tr_idx)

			self.kfcv_err[i] = kfcv_est(model=model,
										X=X_tr, 
										y=y_tr,
										k=k)[0]

			self.spcv_err[i] = spcv_est(model=model,
										X=X_tr, 
										y=y_tr,
										coord=coord_tr,
										k=k)[0]

			self.rela_err[i] = rela_est(X=X, 
										y=y, 
										tr_idx=tr_idx,
										Chol_t=Chol_t*sigma,
										alpha=alpha)

		return self.test_err, self.kfcv_err, self.spcv_err, self.rela_err


	def compareForestTrTs(
		self, 
		niter=100,
		n=200,
		p=30,
		s=5,
		snr=0.4, 
		X=None,
		beta=None,
		coord=None,
		Chol_t=None,
		max_depth=4,
		n_estimators=5,
		tr_idx=None,
		k=10,
		):

		model = RandomForestRegressor(max_depth=max_depth, 
									  n_estimators=n_estimators)

		self.test_err = np.zeros(niter)
		self.kfcv_err = np.zeros(niter)
		self.spcv_err = np.zeros(niter)
		self.rela_err = np.zeros(niter)

		test_est = test_est_split
		kfcv_est = kfoldcv
		spcv_est = kmeanscv
		rela_est = cp_random_forest_train_test
		# rela_est = blur_forest

		# gen_beta = X is None or beta is None

		# if not gen_beta:
		mu = X @ beta
		sigma = np.sqrt(np.var(mu)/snr)
		n, p = X.shape

		tr_idx = np.ones(n).astype(bool)

		for i in np.arange(niter):
		
			# if gen_beta:
			# 	X = np.random.randn(n,p)
			# 	beta = np.zeros(p)
			# 	idx = np.random.choice(p,size=s)
			# 	beta[idx] = np.random.uniform(-1,1,size=s)

			# 	mu = X @ beta
			# 	sigma = np.sqrt(np.var(mu)/snr)

			# tr_idx = create_clus_split(int(np.sqrt(n)), int(np.sqrt(n)))
			# ts_idx = (1 - tr_idx).astype(bool)
			if i == 0:
				print(tr_idx.mean())
			# if tr_idx is None:
				# tr_samples = np.random.choice(n, size=int(.8*n), replace=False)
				# tr_idx = np.zeros(n)
				# tr_idx[tr_samples] = 1
				# ts_idx = (1 - tr_idx).astype(bool)

			eps = sigma * np.random.randn(n)
			eps2 = sigma * np.random.randn(n)
			if Chol_t is not None:
				eps = Chol_t @ eps
				eps2 = Chol_t @ eps2
			y = mu + eps
			y2 = mu + eps2

			X_tr = X[tr_idx,:]
			# X_ts = X[ts_idx,:]
			y_tr = y[tr_idx]
			# y_ts = y[ts_idx]
			coord_tr = coord[tr_idx,:]
			# return X_tr, y_tr, coord_tr

			# self.test_err[i] = test_est(model=model,
			# 							X=X, 
			# 							y=y, 
			# 							y2=y2,
			# 							tr_idx=tr_idx)

			self.kfcv_err[i] = kfcv_est(model=model,
										X=X_tr, 
										y=y_tr,
										k=k)[0]

			self.spcv_err[i] = spcv_est(model=model,
										X=X_tr, 
										y=y_tr,
										coord=coord_tr,
										k=k)[0]

			self.rela_err[i], fitted_model,_ = rela_est(X=X, 
										y=y, 
										tr_idx=tr_idx,
										Chol_t=Chol_t*sigma,
										max_depth=max_depth,
										n_estimators=n_estimators)
			# (self.rela_err[i], fitted_model, w) = rela_est(X, 
			# 											   y, 
			# 											   Chol_t=Chol_t*sigma, 
			# 											   Chol_eps=Chol_t*sigma,
			# 											   model=BlurredForest(max_depth=max_depth,
			# 											   						n_estimators=n_estimators),
			# 											   rand_type='full',
			# 											   use_expectation=True,
			# 											   est_risk=False)

			G = fitted_model.get_group_X(X)
			# y_fit = [y if rand_type == 'full' else w[:,i] for i in np.arange(len(G))]
			# if rand_type == 'full':
			# 	y_fit = [y for _ in np.arange(len(G))]
			# else:
			# 	y_fit = [w[:,i] for i in np.arange(len(G))]
			self.test_err[i] = test_est(model=LinearRegression(fit_intercept=False),
										X=G,
										y=y,
										y2=y2, 
										tr_idx=tr_idx)

		return self.test_err, self.kfcv_err, self.spcv_err, self.rela_err






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
		cbiso_est = cb_isotropic

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

	def compareLinearSelectorIID(self, 
								 niter=100,
								 n=100,
								 p=200,
								 s=5,
								 snr=0.4, 
								 X=None,
								 beta=None,
								 model=Tree(),
								 rand_type='full',
								 use_expectation=False,
								 alpha=0.05,
								 est_sigma=False,
								 est_risk=True):


		self.test_err = np.zeros(niter)
		self.blur_err = np.zeros(niter)

		test_est = test_set_estimator
		blur_est = blur_linear_selector

		gen_beta = X is None or beta is None

		if not gen_beta:
			mu = X @ beta
			sigma_true = np.sqrt(np.var(mu)/snr)
			n, p = X.shape

		for i in np.arange(niter):
		
			if gen_beta:
				X = np.random.randn(n,p)
				beta = np.zeros(p)
				idx = np.random.choice(p,size=s)
				beta[idx] = np.random.uniform(-1,1,size=s)

				mu = X @ beta
				sigma_true = np.sqrt(np.var(mu)/snr)

			y = mu + sigma_true * np.random.randn(n)
			y_test = mu + sigma_true * np.random.randn(n)

			if est_sigma:
				# if isinstance(model, RelaxedLasso):
				# 	model.fit(X,y)
				# 	s = len(model.E_)
				# 	yhat = model.predict(X)
				# 	sigma = np.sum((y-yhat)**2)/(n-s)
				# else:
				# 	model.fit(X, y)
				# 	yhat = model.predict(X)
				# 	sigma = np.std(y - yhat)

				model.fit(X,y)
				P = model.get_linear_smoother(X)
				df = np.diag(P).sum()
				yhat = P @ y
				sigma = np.sum((y - yhat)**2)/(n-df)

			else:
				sigma = sigma_true

			(self.blur_err[i], fitted_model, w) = blur_est(X, 
														   y, 
														   Chol_t=np.eye(n)*sigma, 
														   Chol_eps=np.eye(n)*np.sqrt(alpha)*sigma,
														   model=model,
														   rand_type=rand_type,
														   use_expectation=use_expectation,
														   est_risk=est_risk)

			G = fitted_model.get_group_X(X)
			y_fit = y if rand_type == 'full' else w
			self.test_err[i] = test_est(model=LinearRegression(),
										X=G,
										y=y_fit,
										y_test=y_test, 
										Chol_t=np.eye(n)*sigma, 
										est_risk=est_risk)[0]

		return (self.test_err,
				self.blur_err)

	def compareLinearSelector(self, 
								 niter=100,
								 n=100,
								 p=200,
								 s=5,
								 snr=0.4, 
								 Chol_t=None,
								 Theta_p=None,
								 X=None,
								 beta=None,
								 model=Tree(),
								 rand_type='full',
								 use_expectation=False,
								 alpha=0.05,
								 est_sigma=False,
								 est_risk=True):


		self.test_err = np.zeros(niter)
		self.blur_err = np.zeros(niter)

		test_est = test_set_estimator
		blur_est = blur_linear_selector

		gen_beta = X is None or beta is None

		if not gen_beta:
			mu = X @ beta
			sigma_true = np.sqrt(np.var(mu)/snr)
			n, p = X.shape

		if Chol_t is None:
			Chol_t = np.eye(n)
		if Theta_p is not None:
			Theta_p = Theta_p/(sigma_true**2)

		for i in np.arange(niter):
		
			if gen_beta:
				X = np.random.randn(n,p)
				beta = np.zeros(p)
				idx = np.random.choice(p,size=s)
				beta[idx] = np.random.uniform(-1,1,size=s)

				mu = X @ beta
				sigma_true = np.sqrt(np.var(mu)/snr)

			y = mu + sigma_true * np.random.randn(n)
			y_test = mu + sigma_true * np.random.randn(n)

			if est_sigma:
				# if isinstance(model, RelaxedLasso):
				# 	model.fit(X,y)
				# 	s = len(model.E_)
				# 	yhat = model.predict(X)
				# 	sigma = np.sum((y-yhat)**2)/(n-s)
				# else:
				# 	model.fit(X, y)
				# 	yhat = model.predict(X)
				# 	sigma = np.std(y - yhat)

				model.fit(X,y)
				P = model.get_linear_smoother(X)
				df = np.diag(P).sum()
				yhat = P @ y
				sigma = np.sum((y - yhat)**2)/(n-df)

			else:
				sigma = sigma_true

			(self.blur_err[i], fitted_model, w) = blur_est(X, 
														   y, 
														   Chol_t=Chol_t*sigma, 
														   Chol_eps=Chol_t*np.sqrt(alpha)*sigma,
														   Theta_p=Theta_p,
														   model=model,
														   rand_type=rand_type,
														   use_expectation=use_expectation,
														   est_risk=est_risk)

			G = fitted_model.get_group_X(X)
			y_fit = y if rand_type == 'full' else w
			self.test_err[i] = test_est(model=LinearRegression(),
										X=G,
										y=y_fit,
										y_test=y_test, 
										Chol_t=Chol_t*sigma,
									    # Chol_eps=Chol_t*np.sqrt(alpha)*sigma,
										Theta_p=Theta_p,
										est_risk=est_risk)[0]

		return (self.test_err,
				self.blur_err)

	def compareForestIID(self, 
						 niter=100,
						 n=100,
						 p=200,
						 s=5,
						 snr=0.4, 
						 X=None,
						 beta=None,
						 model=Tree(),
						 rand_type='full',
						 use_expectation=False,
						 alpha=0.05,
						 est_sigma=False,
						 est_risk=True):


		self.test_err = np.zeros(niter)
		self.blur_err = np.zeros(niter)
		# self.tree_err = np.zeros(niter)

		test_est = test_set_estimator
		blur_est = blur_forest
		# tree_est = blur_linear_selector

		gen_beta = X is None or beta is None

		if not gen_beta:
			mu = X @ beta
			sigma_true = np.sqrt(np.var(mu)/snr)
			n, p = X.shape

		for i in np.arange(niter):
		
			if gen_beta:
				X = np.random.randn(n,p)
				beta = np.zeros(p)
				idx = np.random.choice(p,size=s)
				beta[idx] = np.random.uniform(-1,1,size=s)

				mu = X @ beta
				sigma_true = np.sqrt(np.var(mu)/snr)

			y = mu + sigma_true * np.random.randn(n)
			y_test = mu + sigma_true * np.random.randn(n)

			if est_sigma:
				# if isinstance(model, RelaxedLasso):
				# 	model.fit(X,y)
				# 	s = len(model.E_)
				# 	yhat = model.predict(X)
				# 	sigma = np.sum((y-yhat)**2)/(n-s)
				# else:
				# 	model.fit(X, y)
				# 	yhat = model.predict(X)
				# 	sigma = np.std(y - yhat)

				model.fit(X,y)
				P = model.get_linear_smoother(X)
				df = np.diag(P).sum()
				yhat = P @ y
				sigma = np.sum((y - yhat)**2)/(n-df)

			else:
				sigma = sigma_true

			# (self.tree_err[i], fitted_model, w) = tree_est(X, 
			# 											   y, 
			# 											   Chol_t=np.eye(n)*sigma, 
			# 											   Chol_eps=np.eye(n)*np.sqrt(alpha)*sigma,
			# 											   model=Tree(max_depth=4),
			# 											   rand_type=rand_type,
			# 											   use_expectation=use_expectation,
			# 											   est_risk=est_risk)

			(self.blur_err[i], fitted_model, w) = blur_est(X, 
														   y, 
														   # eps=w - y,
														   Chol_t=np.eye(n)*sigma, 
														   Chol_eps=np.eye(n)*np.sqrt(alpha)*sigma,
														   model=model,
														   rand_type=rand_type,
														   use_expectation=use_expectation,
														   est_risk=est_risk)

			G = fitted_model.get_group_X(X)
			y_fit = [y if rand_type == 'full' else w[:,i] for i in np.arange(len(G))]
			# if rand_type == 'full':
			# 	y_fit = [y for _ in np.arange(len(G))]
			# else:
			# 	y_fit = [w[:,i] for i in np.arange(len(G))]
			self.test_err[i] = test_est(model=LinearRegression(),
										X=G,
										y=y_fit,
										y_test=y_test, 
										Chol_t=np.eye(n)*sigma, 
										est_risk=est_risk)[0]

		return (self.test_err,
				# self.tree_err,
				self.blur_err)

	def compareForest(
		self, 
		niter=100,
		n=100,
		p=200,
		s=5,
		snr=0.4, 
		Chol_t=None,
		Theta_p=None,
		X=None,
		beta=None,
		model=Tree(),
		rand_type='full',
		use_expectation=False,
		alpha=0.05,
		est_sigma=False,
		est_risk=True):


		self.test_err = np.zeros(niter)
		self.blur_err = np.zeros(niter)
		# self.tree_err = np.zeros(niter)

		test_est = test_set_estimator
		blur_est = blur_forest
		# tree_est = blur_linear_selector

		gen_beta = X is None or beta is None

		if not gen_beta:
			mu = X @ beta
			sigma_true = np.sqrt(np.var(mu)/snr)
			n, p = X.shape

		if Chol_t is None:
			Chol_t = np.eye(n)
		if Theta_p is not None:
			Theta_p = Theta_p/(sigma_true**2)

		for i in np.arange(niter):
		
			if gen_beta:
				X = np.random.randn(n,p)
				beta = np.zeros(p)
				idx = np.random.choice(p,size=s)
				beta[idx] = np.random.uniform(-1,1,size=s)

				mu = X @ beta
				sigma_true = np.sqrt(np.var(mu)/snr)

			eps = sigma_true * np.random.randn(n)
			eps_test = sigma_true * np.random.randn(n)
			if Chol_t is not None:
				eps = Chol_t @ eps
				eps_test = Chol_t @ eps_test

			y = mu + eps
			y_test = mu + eps_test

			if est_sigma:
				# if isinstance(model, RelaxedLasso):
				# 	model.fit(X,y)
				# 	s = len(model.E_)
				# 	yhat = model.predict(X)
				# 	sigma = np.sum((y-yhat)**2)/(n-s)
				# else:
				# 	model.fit(X, y)
				# 	yhat = model.predict(X)
				# 	sigma = np.std(y - yhat)

				model.fit(X,y)
				P = model.get_linear_smoother(X)
				df = np.diag(P).sum()
				yhat = P @ y
				sigma = np.sum((y - yhat)**2)/(n-df)

			else:
				sigma = sigma_true

			# (self.tree_err[i], fitted_model, w) = tree_est(X, 
			# 											   y, 
			# 											   Chol_t=np.eye(n)*sigma, 
			# 											   Chol_eps=np.eye(n)*np.sqrt(alpha)*sigma,
			# 											   model=Tree(max_depth=4),
			# 											   rand_type=rand_type,
			# 											   use_expectation=use_expectation,
			# 											   est_risk=est_risk)

			(self.blur_err[i], fitted_model, w) = blur_est(X, 
														   y, 
														   # eps=w - y,
														   Chol_t=Chol_t*sigma, 
														   Chol_eps=Chol_t*np.sqrt(alpha)*sigma,
														   Theta_p=Theta_p,
														   model=model,
														   rand_type=rand_type,
														   use_expectation=use_expectation,
														   est_risk=est_risk)

			G = fitted_model.get_group_X(X)
			y_fit = [y if rand_type == 'full' else w[:,i] for i in np.arange(len(G))]
			# if rand_type == 'full':
			# 	y_fit = [y for _ in np.arange(len(G))]
			# else:
			# 	y_fit = [w[:,i] for i in np.arange(len(G))]
			self.test_err[i] = test_est(model=LinearRegression(),
										X=G,
										y=y_fit,
										y_test=y_test, 
										Chol_t=Chol_t*sigma, 
										# Chol_eps=Chol_t*np.sqrt(alpha)*sigma,
										Theta_p=Theta_p,
										est_risk=est_risk)[0]

		return (self.test_err,
				# self.tree_err,
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

	def compareBlurLinear(self, 
							 niter=100,
							 n=100,
							 p=20,
							 s=20,
							 snr=0.4, 
							 Chol_t=None,
							 Theta_p=None,
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

		# print(n*sigma**2)
		# print("------")

		if Chol_t is None:
			Chol_t = np.eye(n)
		if Theta_p is not None:
			Theta_p = Theta_p/(sigma**2)

		for i in np.arange(niter):
		
			if gen_beta:
				X = np.random.randn(n,p)
				beta = np.zeros(p)
				idx = np.random.choice(p,size=s)
				beta[idx] = np.random.uniform(-1,1,size=s)

				mu = X @ beta
				sigma = np.sqrt(np.var(mu)/snr)

			eps = sigma * np.random.randn(n)
			eps_test = sigma * np.random.randn(n)
			eps = Chol_t @ eps
			eps_test = Chol_t @ eps_test

			y = mu + eps
			y_test = mu + eps_test

			y = mu + sigma * np.random.randn(n)
			y_test = mu + sigma * np.random.randn(n)
			y_alpha = mu + sigma * np.sqrt(1 + alpha) * np.random.randn(n)
			y_test_alpha = mu + sigma * np.sqrt(1 + alpha) * np.random.randn(n)

			self.test_err[i] = test_est(model=model,
										X=X, 
										y=y, 
										y_test=y_test, 
										Chol_t=Chol_t*sigma, 
										Theta_p=Theta_p,
										Theta_e=None,
										est_risk=est_risk)[0]
			# self.test_err_alpha[i] = test_est(model=model,
			# 								  X=X, 
			# 								  y=y_alpha, 
			# 								  y_test=y_test_alpha, 
			# 								  Chol_t=Chol_t*np.sqrt(1+alpha)*sigma,
			# 								  Theta_p=Theta_p,
			# 								  est_risk=est_risk)[0]
			# self.cb_err[i] = cb_est(X=X, 
			# 						y=y, 
			# 						Chol_t=Chol_t*sigma, 
			# 						Chol_eps=Chol_t*np.sqrt(alpha)*sigma,
			# 						Theta_p=Theta_p,
			# 						model=model,
			# 						est_risk=est_risk)[0]
			self.blur_err[i] = blur_est(X, 
										y, 
										Chol_t=Chol_t*sigma, 
										Chol_eps=Chol_t*np.sqrt(alpha)*sigma,
										Theta_p=Theta_p,
										Theta_e=None,
										alpha=alpha,
										# Theta_e=np.linalg.inv(Chol_t @ Chol_t.T)/(alpha*sigma**2),
										model=model,
										est_risk=est_risk)[0]

		return (self.test_err,
				self.test_err_alpha,
				self.cb_err,
				self.blur_err)



	








