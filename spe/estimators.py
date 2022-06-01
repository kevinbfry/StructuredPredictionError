import numpy as np
from itertools import product
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate, GroupKFold, KFold
from sklearn.cluster import KMeans

from sklearn.base import clone
from sklearn.utils.validation import check_X_y

from .relaxed_lasso import RelaxedLasso
from .tree import Tree
from .forest import BlurredForest


def _preprocess_X_y_model(X, y, model):
	X, y = check_X_y(X, y)
	(n, p) = X.shape

	if model is not None:
		model = clone(model)

	return X, y, model, n, p


def _get_rand_bool(rand_type):
	return rand_type == 'full'


def _compute_matrices(n, Chol_t, Chol_eps, Theta_p):
	if Chol_eps is None:
		Chol_eps = np.eye(n)
		Sigma_eps = np.eye(n)
	else:
		Sigma_eps = Chol_eps @ Chol_eps.T
	
	Prec_eps = np.linalg.inv(Sigma_eps)

	if Chol_t is None:
		Chol_t = np.eye(n)
		Sigma_t = np.eye(n)
	else:
		Sigma_t = Chol_t @ Chol_t.T

	proj_t_eps = Sigma_t @ Prec_eps

	if Theta_p is None:
		 Theta_p = np.eye(n)
		 Chol_p = np.eye(n)
	else:
		if np.count_nonzero(Theta_p - np.diag(np.diagonal(Theta_p))) == 0:
			Chol_p = np.diag(np.sqrt(np.diagonal(Theta_p)))
		else:
			Chol_p = np.linalg.cholesky(Theta_p)
	Sigma_t_Theta_p = Sigma_t @ Theta_p

	Aperpinv = np.eye(n) + proj_t_eps
	Aperp = np.linalg.inv(Aperpinv)

	return Chol_t, Sigma_t, \
			Chol_eps, Sigma_eps, \
			Prec_eps, proj_t_eps, \
			Theta_p, Chol_p, \
			Sigma_t_Theta_p, Aperp


def _blur(y, Chol_eps, proj_t_eps):
	n = y.shape[0]
	eps = Chol_eps @ np.random.randn(n)
	w = y + eps
	regress_t_eps = proj_t_eps @ eps
	wp = y - regress_t_eps

	return w, wp, eps, regress_t_eps


def _get_covs(Chol_t, Chol_s, alpha=1.):
	n = Chol_t.shape[0]
	if Chol_t is None:
		Chol_t = np.eye(n)
		Sigma_t = np.eye(n)
	else:
		Sigma_t = Chol_t @ Chol_t.T

	same_cov = Chol_s is None
	if same_cov:
		# Chol_s = np.zeros_like(Chol_t)
		# Sigma_s = np.zeros_like(Sigma_t)
		Chol_s = Chol_t
		Sigma_s = Sigma_t
	else:
		Sigma_s = Chol_s @ Chol_s.T

	Chol_eps = np.sqrt(alpha) * Chol_t
	proj_t_eps = np.eye(n) / alpha

	return Chol_t, Sigma_t, Chol_s, Sigma_s, Chol_eps, proj_t_eps






def split_data(
	X, 
	y, 
	tr_idx,
	):

	ts_idx = 1 - tr_idx
	if ts_idx.sum() == 0:
		ts_idx = tr_idx
	tr_idx = tr_idx.astype(bool)
	ts_idx = ts_idx.astype(bool)

	n_tr = tr_idx.sum()
	n_ts = ts_idx.sum()

	X_tr = X[tr_idx,:]
	y_tr = y[tr_idx]

	X_ts = X[ts_idx,:]
	y_ts = y[ts_idx]

	return X_tr, X_ts, y_tr, y_ts, tr_idx, ts_idx, n_tr, n_ts


def better_test_est_split(
	model, 
	X,
	y,
	y2,
	tr_idx,
	**kwargs,
	):

	model = clone(model)
	
	X, y, model, n, p = _preprocess_X_y_model(X, y, model)

	(X_tr, X_ts, y_tr, y_ts, tr_idx, ts_idx, n_tr, n_ts) = split_data(X, y, tr_idx)
	y2_ts = y2[ts_idx]

	model.fit(X_tr, y_tr, **kwargs)
	preds = model.predict(X_ts)

	sse = np.sum((y2_ts - preds)**2)
	return sse / n_ts


def test_est_split(
	model, 
	X,
	y,
	y2,
	tr_idx,
	):

	model = clone(model)

	multiple_X = isinstance(X, list)

	if multiple_X:
		n = X[0].shape[0]
	else:
		n = X.shape[0]


	if multiple_X:
		preds = np.zeros_like(y[0])
		for X_i in X:
			p = X_i.shape[1]
			X_i, y, model, n, p = _preprocess_X_y_model(X_i, y, model)

			(X_i_tr, X_i_ts, y_tr, y_ts, tr_idx, ts_idx, n_tr, n_ts) = split_data(X_i, y, tr_idx)
			y2_ts = y2[ts_idx]

			model.fit(X_i_tr, y_tr)
			preds = model.predict(X_i_ts)

		preds /= len(X)
	else:
		X, y, model, n, p = _preprocess_X_y_model(X, y, model)

		(X_tr, X_ts, y_tr, y_ts, tr_idx, ts_idx, n_tr, n_ts) = split_data(X, y, tr_idx)
		y2_ts = y2[ts_idx]

		model.fit(X_tr, y_tr)
		preds = model.predict(X_ts)

	sse = np.sum((y2_ts - preds)**2)
	return sse / n_ts


def cp_smoother_train_test(
	model,
	X,
	y, 
	tr_idx,
	Chol_t=None,
	Chol_s=None,
	):

	X, y, _, n, p = _preprocess_X_y_model(X, y, None)

	(X_tr, X_ts, y_tr, y_ts, tr_idx, ts_idx, n_tr, n_ts) = split_data(X, y, tr_idx)

	Chol_t, Sigma_t, Chol_s, Sigma_s, _, _ = _get_covs(Chol_t, Chol_s)

	P = X_ts @ np.linalg.inv(X_tr.T @ X_tr) @ X_tr.T
	
	Cov_tr_ts = Sigma_t[tr_idx,:][:,ts_idx]
	Cov_s_ts = Sigma_s[ts_idx,:][:,ts_idx]
	Cov_t_ts = Sigma_t[ts_idx,:][:,ts_idx]

	correction = 2*np.diag(Cov_tr_ts @ P).sum() \
							+ np.diag(Cov_s_ts).sum() \
							- np.diag(Cov_t_ts).sum()

	return (np.sum((y_ts - P @ y_tr)**2) + correction) / n_ts


def cp_linear_train_test(
	model,
	X,
	y, 
	tr_idx,
	Chol_t=None,
	Chol_s=None,
	):
	if model.__class__.__name__ == 'LinearRegression':
		return cp_smoother_train_test(model,
										X,
										y, 
										tr_idx,
										Chol_t,
										Chol_s,
										)
	else:
		return cp_smoother_train_test(LinearRegression(fit_intercept=False),
										X,
										y, 
										tr_idx,
										Chol_t,
										Chol_s,
										)


def cp_adaptive_smoother_train_test(
	model,
	X,
	y, 
	tr_idx,
	Chol_t=None,
	Chol_s=None,
	nboot=100,
	alpha=1.,
	use_trace_corr=True,
	):
	
	model = clone(model)

	X, y, _, n, p = _preprocess_X_y_model(X, y, None)

	(X_tr, X_ts, y_tr, y_ts, tr_idx, ts_idx, n_tr, n_ts) = split_data(X, y, tr_idx)

	Chol_t, Sigma_t, Chol_s, Sigma_s, Chol_eps, proj_t_eps = _get_covs(Chol_t, Chol_s)

	boot_ests = np.zeros(nboot)
	for i in range(nboot):
		w, wp, eps, regress_t_eps = _blur(y, Chol_eps, proj_t_eps)
		w_tr = w[tr_idx]
		wp_ts = wp[ts_idx]

		model.fit(X_tr, w_tr)

		P = model.get_linear_smoother(X_tr, X_ts)
		
		Cov_tr_ts = Sigma_t[tr_idx,:][:,ts_idx]
		Cov_s_ts = Sigma_s[ts_idx,:][:,ts_idx]

		Cov_wp = (1 + 1/alpha)*Sigma_t
		Cov_wp_ts = Cov_wp[ts_idx,:][:,ts_idx]

		correction = 2*np.diag(Cov_tr_ts @ P).sum()
		if use_trace_corr:
			correction += np.diag(Cov_s_ts).sum() \
							- np.diag(Cov_wp_ts).sum()
		else:
			Cov_t_ts = Sigma_t[ts_idx,:][:,ts_idx]
			correction += np.diag(Cov_s_ts).sum() \
							- np.diag(Cov_t_ts).sum() \
							- (regress_t_eps[ts_idx]**2).sum()

		boot_ests[i] = np.sum((wp_ts - P @ y_tr)**2) + correction

	return boot_ests.mean() / n_ts#, model


def cp_relaxed_lasso_train_test(
	model,
	X,
	y, 
	tr_idx,
	Chol_t=None,
	Chol_s=None,
	nboot=100,
	alpha=1.,
	use_trace_corr=True,
	):
	if model.__class__.__name__ == 'RelaxedLasso':
		return cp_adaptive_smoother_train_test(model,
												X,
												y, 
												tr_idx,
												Chol_t,
												Chol_s,
												nboot,
												alpha,
												use_trace_corr,
												)
	else:
		return cp_adaptive_smoother_train_test(RelaxedLasso(lambd=1.),
												X,
												y, 
												tr_idx,
												Chol_t,
												Chol_s,
												nboot,
												alpha,
												use_trace_corr,
												)


def cp_bagged_train_test(
	model,
	X,
	y, 
	tr_idx,
	Chol_t=None,
	Chol_s=None,
	use_trace_corr=False,
	n_estimators=5,
	**kwargs,
	):
	
	model = clone(model)
	kwargs['chol_eps'] = Chol_t
	kwargs['idx_tr'] = tr_idx

	X, y, _, n, p = _preprocess_X_y_model(X, y, None)

	(X_tr, X_ts, y_tr, y_ts, tr_idx, ts_idx, n_tr, n_ts) = split_data(X, y, tr_idx)

	Chol_t, Sigma_t, Chol_s, Sigma_s, _, _ = _get_covs(Chol_t, Chol_s)

	model.fit(X_tr, y_tr, **kwargs)
	
	Ps = model.get_linear_smoother(X_tr, X_ts)
	eps = model.eps_
	
	Cov_tr_ts = Sigma_t[tr_idx,:][:,ts_idx]
	Cov_ts = Sigma_s[ts_idx,:][:,ts_idx]

	Cov_wp = 2*Sigma_t
	Cov_wp_ts = Cov_wp[ts_idx,:][:,ts_idx]

	n_trees = len(Ps)

	tree_ests = np.zeros(n_trees)
	ws = np.zeros((n, n_trees))
	yhats = np.zeros((n_ts, n_trees))

	for i, (P_i, eps_i) in enumerate(zip(Ps, eps)):
		eps_i = eps_i.ravel()
		w = y + eps_i
		# ws[:,i] = w
		regress_t_eps = eps_i
		wp = y - regress_t_eps
		wp_ts = wp[ts_idx]

		correction = 2*np.diag(Cov_tr_ts @ P_i).sum()
		tree_ests[i] = np.sum((wp_ts - P_i @ y_tr)**2) + correction

		yhat = P_i @ y_tr
		yhats[:,i] = yhat

	centered_preds = yhats.mean(axis=1)[:,None] - yhats
	iter_indep_correction = n_trees * (np.diag(Cov_ts).sum() - np.diag(Cov_wp_ts).sum())

	return (tree_ests.sum() + iter_indep_correction - np.sum((centered_preds)**2)) / (n_ts*n_trees)


def cp_rf_train_test(
	model,
	X,
	y, 
	tr_idx,
	Chol_t=None,
	Chol_s=None,
	n_estimators=5,
	ret_gls=False,
	use_trace_corr=True,
	**kwargs,
	):
	
	model = clone(model)
	kwargs['chol_eps'] = Chol_t
	kwargs['idx_tr'] = tr_idx

	X, y, _, n, p = _preprocess_X_y_model(X, y, None)

	(X_tr, X_ts, y_tr, y_ts, tr_idx, ts_idx, n_tr, n_ts) = split_data(X, y, tr_idx)

	Chol_t, Sigma_t, Chol_s, Sigma_s, _, _ = _get_covs(Chol_t, Chol_s)

	model.fit(X_tr, y_tr, **kwargs)
	
	Ps = model.get_linear_smoother(X_tr, X_ts)
	eps = model.eps_
	
	Cov_tr_ts = Sigma_t[tr_idx,:][:,ts_idx]
	Cov_s_ts = Sigma_s[ts_idx,:][:,ts_idx]
	Cov_t_ts = Sigma_t[ts_idx,:][:,ts_idx]

	Cov_wp = 2*Sigma_t
	Cov_wp_ts = Cov_wp[ts_idx,:][:,ts_idx]

	n_trees = len(Ps)

	tree_ests = np.zeros(n_trees)
	ws = np.zeros((n, n_trees))
	yhats = np.zeros((n_ts, n_trees))

	if ret_gls:
		P_gls_s = model.get_linear_smoother(X_tr, X_ts, np.linalg.cholesky(Sigma_t[tr_idx, :][:,tr_idx]))
		tree_gls_ests = np.zeros(n_trees)
		yhats_gls = np.zeros((n_ts, n_trees))

	for i, (P_i, eps_i) in enumerate(zip(Ps, eps)):
		eps_i = eps_i.ravel()
		w = y + eps_i
		# ws[:,i] = w
		regress_t_eps = eps_i
		wp = y - regress_t_eps
		wp_ts = wp[ts_idx]

		correction = 2*np.diag(Cov_tr_ts @ P_i).sum()
		if not use_trace_corr:
			correction -= (regress_t_eps[ts_idx]**2).sum()

		# correction = 2*np.diag(Cov_tr_ts @ P_i).sum()
		tree_ests[i] = np.sum((wp_ts - P_i @ y_tr)**2) + correction
		yhats[:,i] = P_i @ y_tr
		
		if ret_gls:
			P_gls_i = P_gls_s[i]
			gls_correction = 2*np.diag(Cov_tr_ts @ P_gls_i).sum()
			tree_gls_ests[i] = np.sum((wp_ts - P_gls_i @ y_tr)**2) + gls_correction
			yhats_gls[:,i] = P_gls_i @ y_tr

	centered_preds = yhats.mean(axis=1)[:,None] - yhats
	if use_trace_corr:
		iter_indep_correction = n_trees * (np.diag(Cov_s_ts).sum() - np.diag(Cov_wp_ts).sum())
	else:
		iter_indep_correction = n_trees * (np.diag(Cov_s_ts).sum() - np.diag(Cov_t_ts).sum())
	ols_est = (tree_ests.sum() + iter_indep_correction - np.sum((centered_preds)**2)) / (n_ts*n_trees)

	if ret_gls:
		gls_centered_preds = yhats_gls.mean(axis=1)[:,None] - yhats_gls
		return ols_est, (tree_gls_ests.sum() + iter_indep_correction - np.sum((gls_centered_preds)**2)) / (n_ts*n_trees)

	return ols_est









def cb_isotropic(X,
				 y,
				 sigma=None,
				 nboot=100,
				 alpha=1.,
				 model=LinearRegression(),
				 est_risk=True):

	X, y, model, n, p = _preprocess_X_y_model(X, y, model)

	if sigma is None:
		model.fit(X, y)
		pred = model.predict(X)
		sigma = np.sqrt(((y - pred)**2).mean()) # not sure how to get df for general models...

	boot_ests = np.zeros(nboot)

	for b in np.arange(nboot):
		eps = sigma * np.random.randn(n)
		w = y + eps*np.sqrt(alpha)
		wp = y - eps/np.sqrt(alpha)

		model.fit(X, w)
		yhat = model.predict(X)

		boot_ests[b] = np.sum((wp - yhat)**2) - np.sum(eps**2)/alpha

	return boot_ests.mean()/n  + (sigma**2)*(alpha - (1+alpha)*est_risk), model


def cb(X, 
	   y, 
	   Chol_t=None, 
	   Chol_eps=None,
	   Theta_p=None,
	   nboot=100,
	   model=LinearRegression(),
	   est_risk=True):

	X, y, model, n, p = _preprocess_X_y_model(X, y, model)

	Chol_t, Sigma_t, \
	Chol_eps, Sigma_eps, \
	Prec_eps, proj_t_eps, \
	Theta_p, Chol_p, \
	Sigma_t_Theta_p, Aperp = _compute_matrices(n, Chol_t, Chol_eps, Theta_p)

	Sigma_eps_Theta_p = Sigma_eps @ Theta_p

	boot_ests = np.zeros(nboot)

	for b in np.arange(nboot):
		w, wp, eps, regress_t_eps = _blur(y, Chol_eps, proj_t_eps)

		model.fit(X, w)
		yhat = model.predict(X)

		# boot_ests[b] = np.sum((wp - yhat)**2) - (regress_t_eps.T.dot(Theta_p @ regress_t_eps)).sum()
		boot_ests[b] = np.sum((Chol_p@(wp - yhat))**2) - np.sum((Chol_p @ regress_t_eps)**2)

	return (boot_ests.mean() 
			- np.diag(Sigma_t_Theta_p).sum()*est_risk 
			+ np.diag(Sigma_eps_Theta_p).sum()*(1 - est_risk)) / n, model


def blur_linear(X, 
				y, 
				Chol_t=None, 
				Chol_eps=None,
				Theta_p=None,
				Theta_e=None,
				alpha=None,
				nboot=100,
				model=LinearRegression(),
				est_risk=True):

	X, y, model, n, p = _preprocess_X_y_model(X, y, model)

	Chol_t, Sigma_t, \
	Chol_eps, Sigma_eps, \
	Prec_eps, proj_t_eps, \
	Theta_p, Chol_p, \
	Sigma_t_Theta_p, Aperp = _compute_matrices(n, Chol_t, Chol_eps, Theta_p)

	# Theta_e = Prec_eps
	# Chol_e = np.linalg.cholesky(Theta_e)
	if Theta_e is not None:
		Chol_e = np.linalg.cholesky(Theta_e)
	else:
		Theta_e = Prec_eps
		Chol_e = np.linalg.cholesky(Theta_e)
		# Theta_e = np.eye(n)
		# Chol_e = np.eye(n)
	X_e = Chol_e.T @ X

	# P = X @ np.linalg.inv(X.T @ X) @ X.T
	P = X @ np.linalg.inv(X_e.T @ X_e) @ X.T @ Theta_e

	boot_ests = np.zeros(nboot)

	# assert(np.allclose(Sigma_t_Theta_p, np.eye(n)))
	# assert(np.allclose(np.linalg.inv(Theta_p),Sigma_t))
	# assert(np.allclose(proj_t_eps, np.eye(n) / alpha))
	# assert(np.allclose(Sigma_eps @ Theta_p, np.eye(n) * alpha))
	# assert(np.allclose(proj_t_eps @ Sigma_t_Theta_p, np.eye(n) / alpha))
	# assert(np.allclose(P @ Sigma_eps @ P.T @ Theta_p, Sigma_eps @ Theta_p))
	# assert(np.allclose(P @ Sigma_eps @ P.T @ Theta_p, np.eye(n)*alpha))

	for b in np.arange(nboot):
		w, wp, eps, regress_t_eps = _blur(y, Chol_eps, proj_t_eps)

		model.fit(X_e, Chol_e.T @ w)
		# model.fit(X, w)
		yhat = model.predict(X)
		# yhat2 = P @ w
		# assert(np.allclose(yhat, yhat2))

		# boot_ests[b] = np.sum((wp - yhat)**2) - np.sum(regress_t_eps**2) - np.sum((P @ eps)**2)
		# boot_ests[b] = np.sum((wp - yhat)**2) \
		# 				- np.diag(proj_t_eps @ Sigma_t).sum() \
		# 				- np.diag(P @ Sigma_eps @ P.T).sum()

		# boot_ests[b] = np.sum((Chol_p@(wp - yhat))**2) - np.sum((Chol_p@regress_t_eps)**2) - np.sum((Chol_p @ P @ eps)**2)
		# boot_ests[b] = np.sum((wp - yhat).T.dot(Theta_p.dot((wp - yhat)))) \
		# 				- np.sum(regress_t_eps.T.dot(Theta_p.dot(regress_t_eps))) \
		# 				- np.sum((P @ eps).T.dot(Theta_p.dot(P @ eps)))

		boot_ests[b] = np.sum((wp - yhat).T.dot(Theta_p.dot((wp - yhat)))) 

	# print(np.diag(Sigma_t_Theta_p).sum())
	return (boot_ests.mean() 
			- np.diag(proj_t_eps @ Sigma_t_Theta_p).sum() 
			- np.diag(P @ Sigma_eps @ P.T @ Theta_p).sum()
			- np.diag(Sigma_t_Theta_p).sum()*est_risk) / n, model
	# return (boot_ests.mean() - np.diag(Sigma_t).sum()*est_risk) / n, model


def _compute_correction(
	y, 
	w, 
	wp, 
	P, 
	Aperp, 
	regress_t_eps, 
	Theta_p,
	Chol_p,
	Sigma_t_Theta_p, 
	proj_t_eps, 
	full_rand, 
	use_expectation, 
	est_risk,
	):
	yhat = P @ y if full_rand else P @ w
	PAperp = P @ Aperp
	Theta_p_PAperp = Theta_p @ PAperp
	# print(Sigma_t_Theta_p)
	# print(proj_t_eps)

	if use_expectation:
		# boot_est = np.sum((wp - yhat)**2)
		boot_est = np.sum((Chol_p@(wp - yhat))**2)
	else:
		boot_est = np.sum((Chol_p@(wp - yhat))**2) - np.sum((Chol_p@regress_t_eps)**2)
		if full_rand:
		    boot_est += 2*regress_t_eps.T.dot(Theta_p_PAperp.dot(regress_t_eps))

	expectation_correction = 0.
	if full_rand:
		expectation_correction += 2*np.diag(Sigma_t_Theta_p @ PAperp).sum()
	if use_expectation:
		t_epsinv_t = proj_t_eps @ Sigma_t_Theta_p
		# print(t_epsinv_t)
		expectation_correction -= np.diag(t_epsinv_t).sum()
		if full_rand:
			expectation_correction += 2*np.diag(t_epsinv_t @ PAperp).sum()

	return boot_est + expectation_correction \
			- np.diag(Sigma_t_Theta_p).sum()*est_risk, yhat


def blur_linear_selector(X, 
			   y, 
			   Chol_t=None, 
			   Chol_eps=None,
			   Theta_p=None,
			   Theta_e=None,
			   model=RelaxedLasso(),
			   rand_type='full',
			   use_expectation=False,
			   est_risk=True):

	X, y, model, n, p = _preprocess_X_y_model(X, y, model)

	full_rand = _get_rand_bool(rand_type)

	Chol_t, Sigma_t, \
	Chol_eps, Sigma_eps, \
	Prec_eps, proj_t_eps, \
	Theta_p, Chol_p, \
	Sigma_t_Theta_p, Aperp = _compute_matrices(n, Chol_t, Chol_eps, Theta_p)

	w, wp, eps, regress_t_eps = _blur(y, Chol_eps, proj_t_eps)

	if Theta_e is not None:
		Chol_e = np.linalg.cholesky(Theta_e)
	else:
		# Theta_e = Prec_eps
		# Chol_e = np.linalg.cholesky(Theta_e)
		Chol_e = np.eye(n)
	X_e = Chol_e.T @ X

	# model.fit(X, w)
	model.fit(X_e, Chol_e.T @ w)

	P = model.get_linear_smoother(X)

	est, _ = _compute_correction(y, w, wp, P, 
								Aperp, 
								regress_t_eps, 
								Theta_p,
								Chol_p,
								Sigma_t_Theta_p, 
								proj_t_eps, 
								full_rand, 
								use_expectation, 
								est_risk)

	return est / n, model, w


def get_estimate_terms(y, P, eps, Sigma_t, 
						Theta_p, Chol_p,
						Sigma_t_Theta_p,
						proj_t_eps, Aperp, 
						full_rand, 
						use_expectation,
						est_risk):
	n_est = len(P)
	n = y.shape[0]
	tree_ests = np.zeros(n_est)
	ws = np.zeros((n, n_est))
	yhats = np.zeros((n, n_est))
	for i, (P_i, eps_i) in enumerate(zip(P, eps)):
		eps_i = eps_i.ravel()
		w = y + eps_i
		regress_t_eps = proj_t_eps @ eps_i
		wp = y - regress_t_eps

		tree_ests[i], yhat = _compute_correction(y, w, wp, P_i, 
												Aperp, 
												regress_t_eps, 
												Theta_p,
												Chol_p,
												Sigma_t_Theta_p, 
												proj_t_eps, 
												full_rand, 
												use_expectation, 
												est_risk)

		ws[:,i] = w
		yhats[:,i] = yhat

	centered_preds = yhats.mean(axis=1)[:,None] - yhats

	return tree_ests, centered_preds, ws


def blur_forest(X, 
			    y, 
			    eps=None,
			    Chol_t=None, 
			    Chol_eps=None,
			    Theta_p=None,
			    Theta_e=None,
			    model=BlurredForest(),
			    rand_type='full',
			    use_expectation=False,
			    est_risk=True):

	X, y, model, n, p = _preprocess_X_y_model(X, y, model)

	full_rand = _get_rand_bool(rand_type)

	Chol_t, Sigma_t, \
	Chol_eps, Sigma_eps, \
	Prec_eps, proj_t_eps, \
	Theta_p, Chol_p, \
	Sigma_t_Theta_p, Aperp = _compute_matrices(n, Chol_t, Chol_eps, Theta_p)

	if Theta_e is not None:
		Chol_e = np.linalg.cholesky(Theta_e)
	else:
		# Theta_e = Prec_eps
		# Chol_e = np.linalg.cholesky(Theta_e)
		Chol_e = np.eye(n)
	X_e = Chol_e.T @ X

	model.fit(X_e, y, chol_eps=Chol_eps, bootstrap_type='blur')

	P = model.get_linear_smoother(X)
	if eps is None:
		eps = model.eps_
	else:
		eps = [eps]
	
	n_trees = len(P)

	tree_ests, centered_preds, ws = get_estimate_terms(y, P, eps, Sigma_t, 
														Theta_p, Chol_p,
														Sigma_t_Theta_p,
														proj_t_eps, Aperp, 
														full_rand, 
														use_expectation,
														est_risk)
	return (tree_ests.sum()
			# - np.sum(centered_preds**2)) / (n * n_trees), model, ws
			- np.sum((Chol_e@centered_preds)**2)) / (n * n_trees), model, ws


def bag_kfoldcv(model, 
					X, 
					y, 
					k=10,
					Chol_t=None,
					):

	model = clone(model)
	X, y, model, n, p = _preprocess_X_y_model(X, y, model)

	kf=KFold(k, shuffle=True)

	err = []
	kwargs = dict() if type(model).__name__ == 'BaggedRelaxedLasso' else dict(bootstrap_type='blur')
	for tr_idx, ts_idx in kf.split(X):
		tr_bool = np.zeros(n)
		tr_bool[tr_idx] = 1
		(X_tr, X_ts, y_tr, y_ts, tr_idx, ts_idx, n_tr, n_ts) = split_data(X, y, tr_bool)
		if Chol_t is None:
			kwargs['chol_eps'] = None
		else:
			kwargs['chol_eps'] = Chol_t[tr_idx,:][:,tr_idx]
		model.fit(X_tr, y_tr, **kwargs)
		err.append(np.mean((y_ts - model.predict(X_ts))**2))

	return np.mean(err)

	# err = []
	# for tr_idx, ts_idx in kf.split(X):
	# 	tr_bool = np.zeros(n)
	# 	tr_bool[tr_idx] = 1
	# 	(X_tr, X_ts, y_tr, y_ts, tr_idx, ts_idx, n_tr, n_ts) = split_data(X, y, tr_bool)
	# 	if Chol_t is None:
	# 		Chol_eps = None
	# 	else:
	# 		Chol_eps = Chol_t[tr_idx,:][:,tr_idx]
	# 	model.fit(X_tr, y_tr, chol_eps=Chol_eps)
	# 	err.append(np.mean((y_ts - model.predict(X_ts))**2))

	# return np.mean(err)


def bag_kmeanscv(model, 
					X, 
					y, 
					coord,
					k=10,
					Chol_t=None,
					):

	model = clone(model)
	X, y, model, n, p = _preprocess_X_y_model(X, y, model)
	
	groups = KMeans(n_clusters=k).fit(coord).labels_
	gkf=GroupKFold(k)

	err = []
	kwargs = dict() if type(model).__name__ == 'BaggedRelaxedLasso' else dict(bootstrap_type='blur')
	for tr_idx, ts_idx in gkf.split(X, groups=groups):
		tr_bool = np.zeros(n)
		tr_bool[tr_idx] = 1
		(X_tr, X_ts, y_tr, y_ts, tr_idx, ts_idx, n_tr, n_ts) = split_data(X, y, tr_bool)
		if Chol_t is None:
			kwargs['chol_eps'] = None
		else:
			kwargs['chol_eps'] = Chol_t[tr_idx,:][:,tr_idx]
		model.fit(X_tr, y_tr, **kwargs)
		err.append(np.mean((y_ts - model.predict(X_ts))**2))

	return np.mean(err)

	# err = []
	# for tr_idx, ts_idx in gkf.split(X, groups=groups):
	# 	tr_bool = np.zeros(n)
	# 	tr_bool[tr_idx] = 1
	# 	(X_tr, X_ts, y_tr, y_ts, tr_idx, ts_idx, n_tr, n_ts) = split_data(X, y, tr_bool)
	# 	if Chol_t is None:
	# 		Chol_eps = None
	# 	else:
	# 		Chol_eps = Chol_t[tr_idx,:][:,tr_idx]
	# 	model.fit(X_tr, y_tr, chol_eps=Chol_eps)
	# 	err.append(np.mean((y_ts - model.predict(X_ts))**2))

	# return np.mean(err)


def kfoldcv(model, 
			X, 
			y, 
			k=10,
			**kwargs):

	model = clone(model)

	kfcv_res = cross_validate(model, X, y, 
							scoring='neg_mean_squared_error', 
							cv=KFold(k, shuffle=True), 
							error_score='raise',
							fit_params=kwargs)
	return -np.mean(kfcv_res['test_score'])#, model


def kmeanscv(model, 
			 X, 
			 y, 
			 coord,
			 k=10,
			 **kwargs):

	groups = KMeans(n_clusters=k).fit(coord).labels_
	spcv_res = cross_validate(model, 
								X, 
								y, 
								scoring='neg_mean_squared_error', 
								cv=GroupKFold(k), 
								groups=groups,
								fit_params=kwargs)

	return -np.mean(spcv_res['test_score'])#, model


def test_set_estimator(model, 
					   X, 
					   y,
					   y_test,
					   Chol_t=None,
					   Chol_eps=None,
					   Theta_p=None,
					   Theta_e=None,
					   est_risk=True):

	model = clone(model)

	multiple_X = isinstance(X, list)

	if multiple_X:
		n = X[0].shape[0]

	else:
		n = X.shape[0]

	Chol_t, Sigma_t, \
	Chol_eps, Sigma_eps, \
	Prec_eps, proj_t_eps, \
	Theta_p, Chol_p, \
	Sigma_t_Theta_p, Aperp = _compute_matrices(n, Chol_t, Chol_eps, Theta_p)

	# if Chol_t is None:
	# 	Chol_t = np.eye(n)

	# Sigma_t = Chol_t @ Chol_t.T

	# if Theta_p is None:
	# 	Theta_p = np.eye(n)
	# 	Chol_p = np.eye(n)
	# else:
	# 	if np.count_nonzero(Theta_p - np.diag(np.diagonal(Theta_p))) == 0:
	# 		Chol_p = np.diag(np.sqrt(np.diagonal(Theta_p)))
	# 	else:
	# 		Chol_p = np.linalg.cholesky(Theta_p)

	# Sigma_t_Theta_p = Sigma_t @ Theta_p

	if Theta_e is not None:
		Chol_e = np.linalg.cholesky(Theta_e)
	else:
		# Theta_e = Prec_eps
		# Chol_e = np.linalg.cholesky(Theta_e)
		Chol_e = np.eye(n)

	if multiple_X:
		preds = np.zeros_like(y[0])
		for X_i, y_i in zip(X, y):
			model.fit(Chol_e.T @ X_i, Chol_e.T @ y_i)
			preds += model.predict(Chol_e.T @ X_i)

		preds /= len(X)
	else:
		model.fit(Chol_e.T @ X, Chol_e.T @ y)
		# preds = model.predict(Chol_e.T @ X)
		preds = model.predict(X)
	
	# sse = np.sum((y_test - preds)**2)
	# print(Theta_p)
	# print(Chol_p)
	sse = np.sum((Chol_p @ (y_test - preds))**2)
	return (sse - np.diag(Sigma_t_Theta_p).sum()*est_risk) / n, model













