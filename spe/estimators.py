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
	model = clone(model)
	(n, p) = X.shape

	return X, y, model, n, p


def _get_rand_bool(rand_type):
	return rand_type == 'full'


def _compute_matrices(n, Chol_t, Chol_eps):
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

	Aperpinv = np.eye(n) + proj_t_eps
	Aperp = np.linalg.inv(Aperpinv)

	return Chol_t, Sigma_t, \
			Chol_eps, Sigma_eps, \
			Prec_eps, proj_t_eps, \
			Sigma_t_Theta, Aperp

def _blur(y, Chol_eps, proj_t_eps):
	n = y.shape[0]
	eps = Chol_eps @ np.random.randn(n)
	w = y + eps
	regress_t_eps = proj_t_eps @ eps
	wp = y - regress_t_eps

	return w, wp, eps, regress_t_eps


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

	# return boot_ests.mean()/n - (sigma**2)*est_risk


def cb(X, 
	   y, 
	   Chol_t=None, 
	   Chol_eps=None,
	   Theta=None,
	   nboot=100,
	   model=LinearRegression(),
	   est_risk=True):

	X, y, model, n, p = _preprocess_X_y_model(X, y, model)

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

	if Theta is None:
		Theta = np.eye(n)
	Sigma_t_Theta = Sigma_t @ Theta
	Sigma_eps_Theta = Sigma_eps @ Theta

	boot_ests = np.zeros(nboot)

	for b in np.arange(nboot):
		w, wp, eps, regress_t_eps = _blur(y, Chol_eps, proj_t_eps)

		model.fit(X, w)
		yhat = model.predict(X)

		boot_ests[b] = np.sum((wp - yhat)**2) - (regress_t_eps.T.dot(Theta @ regress_t_eps)).sum()

	return (boot_ests.mean() - np.diag(Sigma_t_Theta).sum()*est_risk + np.diag(Sigma_eps_Theta).sum()*(1 - est_risk)) / n, model



def blur_linear(X, 
				y, 
				Chol_t=None, 
				Chol_eps=None,
				# Theta=None,
				nboot=100,
				model=LinearRegression(),
				est_risk=True):

	X, y, model, n, p = _preprocess_X_y_model(X, y, model)

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
	#   Theta = np.eye(n)
	Sigma_t_Theta = Sigma_t# @ Theta

	P = X @ np.linalg.inv(X.T @ X) @ X.T

	boot_ests = np.zeros(nboot)

	for b in np.arange(nboot):
		w, wp, eps, regress_t_eps = _blur(y, Chol_eps, proj_t_eps)

		model.fit(X, w)
		yhat = model.predict(X)

		boot_ests[b] = np.sum((wp - yhat)**2) - np.sum(regress_t_eps**2) - np.sum((P @ eps)**2)

	return (boot_ests.mean() - np.diag(Sigma_t_Theta).sum()*est_risk) / n, model


def blur_linear_selector(X, 
			   y, 
			   Chol_t=None, 
			   Chol_eps=None,
			   model=RelaxedLasso(),
			   rand_type='full',
			   use_expectation=False,
			   est_risk=True):

	X, y, model, n, p = _preprocess_X_y_model(X, y, model)

	full_rand = _get_rand_bool(rand_type)

	Chol_t, Sigma_t, \
	Chol_eps, Sigma_eps, \
	Prec_eps, proj_t_eps, \
	Sigma_t_Theta, Aperp = _compute_matrices(n, Chol_t, Chol_eps)

	w, wp, eps, regress_t_eps = _blur(y, Chol_eps, proj_t_eps)

	model.fit(X, w)

	P = model.get_linear_smoother(X)
	yhat = P @ y if full_rand else P @ w
	PAperp = P @ Aperp
	# print(P)

	if use_expectation:
		boot_est = np.sum((wp - yhat)**2)
	else:
		boot_est = np.sum((wp - yhat)**2) - np.sum(regress_t_eps**2)
		if full_rand:
		    boot_est += 2*regress_t_eps.T.dot(PAperp.dot(regress_t_eps))

	expectation_correction = 0.
	if full_rand:
		expectation_correction += 2*np.diag(Sigma_t @ PAperp).sum()
	if use_expectation:
		t_epsinv_t = proj_t_eps @ Sigma_t
		expectation_correction -= np.diag(t_epsinv_t).sum()
		if full_rand:
			expectation_correction += 2*np.diag(t_epsinv_t @ PAperp).sum()
	
	return (boot_est + expectation_correction
			- np.diag(Sigma_t_Theta).sum()*est_risk) / n, model, w


def get_estimate_terms(y, P, eps, Sigma_t,
						Sigma_t_Theta,
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

		yhat = P_i @ y if full_rand else P_i @ w
		PAperp = P_i @ Aperp

		if use_expectation:
			boot_est = np.sum((wp - yhat)**2)
		else:
			boot_est = np.sum((wp - yhat)**2) - np.sum(regress_t_eps**2)
			if full_rand:
			    boot_est += 2*regress_t_eps.T.dot(PAperp.dot(regress_t_eps))

		expectation_correction = 0.
		if full_rand:
			expectation_correction += 2*np.diag(Sigma_t @ PAperp).sum()
		if use_expectation:
			t_epsinv_t = proj_t_eps @ Sigma_t
			expectation_correction -= np.diag(t_epsinv_t).sum()
			if full_rand:
				expectation_correction += 2*np.diag(t_epsinv_t @ PAperp).sum()
		
		tree_ests[i] = boot_est + expectation_correction - np.diag(Sigma_t_Theta).sum()*est_risk
		ws[:,i] = w
		yhats[:,i] = yhat

	centered_preds = yhats.mean(axis=1)[:,None] - yhats

	return tree_ests, centered_preds, ws


def blur_forest(X, 
			    y, 
			    eps=None,
			    Chol_t=None, 
			    Chol_eps=None,
			    model=BlurredForest(),
			    rand_type='full',
			    use_expectation=False,
			    est_risk=True):

	X, y, model, n, p = _preprocess_X_y_model(X, y, model)

	full_rand = _get_rand_bool(rand_type)

	Chol_t, Sigma_t, \
	Chol_eps, Sigma_eps, \
	Prec_eps, proj_t_eps, \
	Sigma_t_Theta, Aperp = _compute_matrices(n, Chol_t, Chol_eps)

	# model.fit(X, y+eps, chol_eps=np.zeros_like(Chol_eps))
	model.fit(X, y, chol_eps=Chol_eps)#, Sigma_t=Sigma_t)

	P = model.get_linear_smoother(X)
	if eps is None:
		eps = model.eps_
	else:
		eps = [eps]
	
	tree_ests, centered_preds, ws = get_estimate_terms(y, P, eps, Sigma_t,
														Sigma_t_Theta,
														proj_t_eps, Aperp, 
														full_rand, 
														use_expectation,
														est_risk)
	# print(tree_ests.sum(), np.sum(centered_preds**2))
	return (tree_ests.sum() 
			- np.sum(centered_preds**2)) / n, model, ws
	# return (tree_ests.sum()) / n, model, ws


def kfoldcv(model, 
			X, 
			y, 
			k=5):

	model = clone(model)

	kfcv_res = cross_validate(model, X, y, 
							scoring='neg_mean_squared_error', 
							cv=KFold(k, shuffle=True), 
							error_score='raise')
	return -np.mean(kfcv_res['test_score']), model



def kmeanscv(model, 
			 X, 
			 y, 
			 k=5):

	groups = KMeans(n_clusters=k).fit(X).labels_
	spcv_res = spcv_res = cross_validate(model, 
											X, 
											y, 
											scoring='neg_mean_squared_error', 
											cv=GroupKFold(k), 
											groups=groups)

	return -np.mean(spcv_res['test_score']), model



def test_set_estimator(model, 
					   X, 
					   y,
					   y_test,
					   Chol_t=None, 
					   Theta=None,
					   est_risk=True):

	model = clone(model)

	multiple_X = isinstance(X, list)

	if multiple_X:
		n = X[0].shape[0]

	else:
		n = X.shape[0]

	if Chol_t is None:
		Chol_t = np.eye(n)

	Sigma_t = Chol_t @ Chol_t.T

	if Theta is None:
		Theta = np.eye(n)

	Sigma_t_Theta = Sigma_t @ Theta

	if multiple_X:
		preds = np.zeros_like(y[0])
		for X_i, y_i in zip(X, y):
			model.fit(X_i, y_i)
			preds += model.predict(X_i)

		preds /= len(X)
	else:
		model.fit(X, y)
		preds = model.predict(X)
	
	sse = np.sum((y_test - preds)**2)
	return (sse - np.diag(Sigma_t_Theta).sum()*est_risk) / n, model




