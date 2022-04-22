import numpy as np
from itertools import product
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate, GroupKFold, KFold
from sklearn.cluster import KMeans

from sklearn.base import BaseEstimator, clone
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from .relaxed_lasso import RelaxedLasso


def cb_isotropic(X,
				 y,
				 sigma=None,
				 nboot=100,
				 alpha=1.,
				 model=LinearRegression(),
				 est_risk=True):

	model = clone(model)

	X = X
	y = y
	(n, p) = X.shape

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

	model = clone(model)

	X = X
	y = y
	(n, p) = X.shape

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
		eps = Chol_eps @ np.random.randn(n)
		w = y + eps
		regress_t_eps = proj_t_eps @ eps
		wp = y - regress_t_eps

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

	model = clone(model)

	X = X
	y = y
	(n, p) = X.shape

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
		eps = Chol_eps @ np.random.randn(n)
		w = y + eps
		regress_t_eps = proj_t_eps @ eps
		wp = y - regress_t_eps

		model.fit(X, w)
		yhat = model.predict(X)

		boot_ests[b] = np.sum((wp - yhat)**2) - np.sum(regress_t_eps**2) - np.sum((P @ eps)**2)

	return (boot_ests.mean() - np.diag(Sigma_t_Theta).sum()*est_risk) / n, model



## only full refit for now
  
def blur_lasso(X, 
			   y, 
			   Chol_t=None, 
			   Chol_eps=None,
			   model=RelaxedLasso(),
			   rand_type='full',
			   use_expectation=False,
			   est_risk=True):

	model = clone(model)

	full_rand = rand_type == 'full'

	X = X
	y = y
	(n, p) = X.shape

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

	Aperpinv = np.eye(proj_t_eps.shape[0]) + proj_t_eps
	Aperp = np.linalg.inv(Aperpinv)

	# boot_ests = np.zeros(nboot)

	# for b in np.arange(nboot):
	eps = Chol_eps @ np.random.randn(n)
	w = y + eps
	regress_t_eps = proj_t_eps @ eps
	wp = y - regress_t_eps

	lin_y = y if full_rand else w
	model.fit(X, 
			  lasso_y=w, 
			  lin_y=lin_y)
	yhat = model.predict(X)

	# XE = model.predXE_
	# P = XE @ np.linalg.inv(XE.T @ XE) @ XE.T
	P = model.get_linear_smoother(X)
	PAperp = P @ Aperp

	# boot_est = np.sum((wp - yhat)**2)

	# t_epsinv_t = proj_t_eps @ Sigma_t
	# expectation_correction = - np.diag(t_epsinv_t).sum()
	# if full_rand:
	# 	expectation_correction += 2*np.diag((Sigma_t + t_epsinv_t) @ PAperp).sum()
	
	# return (boot_est + expectation_correction
	# 		- np.diag(Sigma_t_Theta).sum()*est_risk) / n, model, w

	if use_expectation:
		boot_est = np.sum((wp - yhat)**2)
	else:
		boot_est = np.sum((wp - yhat)**2) - np.sum(regress_t_eps**2)
		if full_rand:
		    boot_est += 2*regress_t_eps.T.dot(PAperp.dot(regress_t_eps))

	t_epsinv_t = proj_t_eps @ Sigma_t
	expectation_correction = 0.
	if full_rand:
		expectation_correction += 2*np.diag(Sigma_t @ PAperp).sum()
	if use_expectation:
		expectation_correction -= np.diag(t_epsinv_t).sum()
		if full_rand:
			expectation_correction += 2*np.diag(t_epsinv_t @ PAperp).sum()
	
	return (boot_est + expectation_correction
			- np.diag(Sigma_t_Theta).sum()*est_risk) / n, model, w



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

	(n, p) = X.shape

	if Chol_t is None:
		Chol_t = np.eye(n)

	Sigma_t = Chol_t @ Chol_t.T

	if Theta is None:
		Theta = np.eye(n)

	Sigma_t_Theta = Sigma_t @ Theta

	model.fit(X, y)
	preds = model.predict(X)
	sse = np.sum((y_test - preds)**2)

	# print(np.diag(Sigma_t_Theta).sum())

	return (sse - np.diag(Sigma_t_Theta).sum()*est_risk) / n, model




