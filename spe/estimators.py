"""
TODO:
- Clean up CV methods
- Move est_sigma functionality to spe/
- notebooks:
    - Linear: OLS, Spline, ind/cov
    - Adpative linear: RL, Tree,
    - Bagged: RF, ?, ind/cov
    - Comparison(s) to BY
    - Comparison(s) to CV
    - Compare Oracle to Estimated
- Implement BY estimator
"""

import numpy as np

from itertools import product
from sklearn.linear_model import LinearRegression
# from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate, GroupKFold, KFold, TimeSeriesSplit
from sklearn.cluster import KMeans

from sklearn.base import clone
from sklearn.utils.validation import check_X_y, check_is_fitted

from .relaxed_lasso import RelaxedLasso
from .tree import Tree, LinearSelector
from .forest import BlurredForest, ParametricBaggingRegressor


## TODO: add Chol_eps option, not always do Chol_eps=alpha*Chol_t


def _preprocess_X_y_model(X, y, model):
    X, y = check_X_y(X, y)
    (n, p) = X.shape

    if model is not None:
        model = clone(model)

    return X, y, model, n, p


def _get_rand_bool(rand_type):
    return rand_type == "full"


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

    return (
        Chol_t,
        Sigma_t,
        Chol_eps,
        Sigma_eps,
        Prec_eps,
        proj_t_eps,
        Theta_p,
        Chol_p,
        Sigma_t_Theta_p,
        Aperp,
    )


def _blur(y, Chol_eps, proj_t_eps=None):
    n = y.shape[0]
    eps = Chol_eps @ np.random.randn(n)
    w = y + eps
    if proj_t_eps is not None:
        regress_t_eps = proj_t_eps @ eps
        wp = y - regress_t_eps

        return w, wp, eps, regress_t_eps
    
    return w, eps


def _get_covs(Chol_t, Chol_s, alpha=1.0):
    n = Chol_t.shape[0]
    if Chol_t is None:
        Chol_t = np.eye(n)
        Sigma_t = np.eye(n)
    else:
        Sigma_t = Chol_t @ Chol_t.T

    same_cov = Chol_s is None
    if same_cov:
        Chol_s = Chol_t
        Sigma_s = Sigma_t
    else:
        Sigma_s = Chol_s @ Chol_s.T

    ## TODO: change when allowing general Chol_eps
    Chol_eps = np.sqrt(alpha) * Chol_t
    proj_t_eps = np.eye(n) / alpha

    return Chol_t, Sigma_t, Chol_s, Sigma_s, Chol_eps, proj_t_eps


def split_data(
    X,
    y,
    tr_idx,
):

    tr_idx = tr_idx.astype(bool)
    ts_idx = ~tr_idx#1 - tr_idx
    if ts_idx.sum() == 0:
        ts_idx = tr_idx
    else:
        assert(np.sum(tr_idx * ts_idx) == 0)
        assert(np.sum(tr_idx + ts_idx) == X.shape[0])

    tr_idx = tr_idx.astype(bool)
    ts_idx = ts_idx.astype(bool)

    n_tr = tr_idx.sum()
    n_ts = ts_idx.sum()

    X_tr = X[tr_idx, :]
    y_tr = y[tr_idx]

    X_ts = X[ts_idx, :]
    y_ts = y[ts_idx]

    return X_tr, X_ts, y_tr, y_ts, tr_idx, ts_idx, n_tr, n_ts


def better_test_est_split(
    model,
    X,
    y,
    y2,
    tr_idx,
    full_refit=False,
    alpha=None,
    Chol_t=None,
    gls=None,
    bagg=False,
    # chol=None,
    **kwargs,
):
    X, y, model, n, p = _preprocess_X_y_model(X, y, model)

    (X_tr, X_ts, y_tr, y_ts, tr_idx, ts_idx, n_tr, n_ts) = split_data(X, y, tr_idx)
    y2_ts = y2[ts_idx]
    # y2_ts = y_ts

    if bagg:
        base_model = clone(model)
        model = ParametricBaggingRegressor(base_estimator=base_model, n_estimators=100)

    if alpha is not None:
        w, eps = _blur(y, np.sqrt(alpha) * Chol_t)
        w_tr = w[tr_idx]
        model.fit(X_tr, w_tr, **kwargs)
    else:
        if isinstance(model, (ParametricBaggingRegressor, BlurredForest)):
            if "chol_eps" in kwargs:
                pass
            else:
                kwargs["chol_eps"] = Chol_t
            kwargs["idx_tr"] = tr_idx
        model.fit(X_tr, y_tr, **kwargs)

    if full_refit is None or full_refit:
        if isinstance(model, (Tree, RelaxedLasso)):
            P = model.get_linear_smoother(X, tr_idx, ts_idx)
            preds = P @ y_tr
        elif isinstance(model, BlurredForest):
            chol = None if not gls else np.linalg.inv(np.linalg.cholesky(
                                            (Chol_t @ Chol_t.T)[tr_idx,:][:,tr_idx]
                                        )).T
            preds = model.predict(X, tr_idx, ts_idx, y_tr)
        elif isinstance(model, ParametricBaggingRegressor):
            preds = model.predict(X, tr_idx, ts_idx, y_tr)
        else:
            preds = model.predict(X_ts)

    else:
        preds = model.predict(X_ts)

    # assert(np.mean((y2_ts - preds)**2) >= 0)
    return np.mean((y2_ts - preds)**2)
    # return np.sum((y2_ts - preds)**2)



## Spatial extension of Breiman-Ye estimator
## \| Y - g(Y) \|_2^2 + 2/\alpha * 1/(B-1)\sum_i (Y^b_i - \bar Y_B)g(Y^b_i) - 2/\alpha * 1/(B-1)\sum_i (Y^(*b_i) - \bar Y^*_B)g(Y^b_i)
## assumes Y^* and Y have same covariance matrix (so no noise correction terms)
def by_spatial(
    model,
    X,
    y,
    alpha,
    Chol_f,
    nboot=100,
    tr_idx=None,
):
    # print("HERE", alpha)
    # assert(0==1)
    X, y, model, n, p = _preprocess_X_y_model(X, y, model)

    assert(Chol_f.shape[0] == Chol_f.shape[1] == 2*n)
    Sigma_f = Chol_f @ Chol_f.T
    assert(np.allclose(Sigma_f[:n,:][:,:n], Sigma_f[n:,:][:,n:]))

    boot_samples = np.zeros((n, nboot))
    boot_star_samples = np.zeros((n, nboot))
    boot_preds = np.zeros((n, nboot))
    for b in np.arange(nboot):
        omega_full = np.sqrt(alpha)*Chol_f @ np.random.randn(2*n)
        omega_1 = omega_full[:n]
        omega_2 = omega_full[n:]
        yb = y + omega_1
        boot_samples[:,b] = yb


        ystarb = y + omega_2
        boot_star_samples[:,b] = ystarb

        model.fit(X, yb)
        preds = model.predict(X)
        boot_preds[:,b] = preds

    model.fit(X,y)
    # print(np.sum(model.coef_ != 0))
    obs_preds = model.predict(X)

    # print(Chol_f[0,0])
    # print(np.mean((boot_samples - boot_samples.mean(1)[:, None]) * boot_preds) / alpha)
    # print(np.mean((boot_star_samples - boot_star_samples.mean(1)[:, None]) * boot_preds) / alpha)
    # boot_cov = np.sum((boot_samples - boot_samples.mean(axis=1)[:, None]) * boot_preds) / (nboot - 1.)
    boot_cov = np.mean((boot_samples - boot_samples.mean(axis=1)[:, None]) * boot_preds)
    boot_star_cov = np.mean((boot_star_samples - boot_star_samples.mean(axis=1)[:, None]) * boot_preds)
    # assert(np.fabs(boot_star_cov) < 1e-4)
    boot_corr = 2 / alpha * (
        boot_cov
        - boot_star_cov
        # np.sum((boot_samples - boot_samples.mean(1)[:, None]) * boot_preds) / (nboot - 1.)
        # - np.sum((boot_star_samples - boot_star_samples.mean(1)[:, None]) * boot_preds) / (nboot - 1.)
    )

    return np.mean((y - obs_preds)**2) + boot_corr#/n 
    # return np.sum((y - obs_preds)**2) + boot_corr#/n 


def ts_test_est_split(
    model,
    X,
    y,
    y2,
    tr_idx,
    full_refit=False,
    alpha=None,
    Chol_t=None,
    gls=None,
    max_train_size=None,
    # chol=None,
    **kwargs,
):
    # print("Chol_t", Chol_t)

    X, y, model, n, p = _preprocess_X_y_model(X, y, model)

    (X_tr, X_ts, y_tr, y_ts, tr_idx, ts_idx, n_tr, n_ts) = split_data(X, y, tr_idx)
    # y2_ts = y2[ts_idx] ## TODO: give option for y2 vs y
    y2_ts = y_ts

    X_tr = X_tr[-max_train_size:,]
    y_tr = y_tr[-max_train_size:]

    # model.fit(X_tr, y_tr, **kwargs)
    # preds = model.predict(X_ts)
    # preds = model.predict(X_ts, **kwargs)

    if alpha is not None:
        w, eps = _blur(y, np.sqrt(alpha) * Chol_t)
        w_tr = w[tr_idx]
        model.fit(X_tr, w_tr, **kwargs)
    else:
        if model.__class__.__name__ == "BlurredForest":
            if "chol_eps" in kwargs:
                kwargs["chol_eps"] = kwargs["chol_eps"][tr_idx, :][:, tr_idx]
        model.fit(X_tr, y_tr, **kwargs)

    if full_refit is None or full_refit:
        if model.__class__.__name__ == "RelaxedLasso":
            P = model.get_linear_smoother(X, tr_idx, ts_idx)#X_tr, X_ts)
            preds = P @ y_tr
        else:
            if gls is None or not gls:
                # print("gls None", gls)
                preds = model.predict(X_ts, full_refit=full_refit)
            else:
                # print("gls not None", gls)
                Chol_t_inv_tr = np.linalg.inv(np.linalg.cholesky(
                    (Chol_t @ Chol_t.T)[tr_idx,:][:,tr_idx]
                )).T
                # print("gls", gls)
                # print("Chol", Chol_t_tr)
                # print("Sigma", Chol_t_tr @ Chol_t_tr.T)
                preds = model.predict(
                    X_ts, 
                    full_refit=full_refit, 
                    Chol=Chol_t_inv_tr
                )
        # preds = model.predict(X_ts, full_refit=full_refit, chol=chol)
    else:
        ## TODO: all the conditional stuff I have for full refit
        preds = model.predict(X_ts)

    # sse = np.sum((y2_ts - preds) ** 2)
    # return sse / n_ts
    return np.mean((y2_ts - preds) ** 2)


def test_est_split(
    model,
    X,
    y,
    y2,
    tr_idx,
    **kwargs,
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
            X_i, y, _, n, p = _preprocess_X_y_model(X_i, y, None)

            (X_i_tr, X_i_ts, y_tr, y_ts, tr_idx, ts_idx, n_tr, n_ts) = split_data(
                X_i, y, tr_idx
            )
            y2_ts = y2[ts_idx]

            # model.fit(X_i_tr, y_tr)
            model.fit(X_i_tr, y_tr, **kwargs)
            preds = model.predict(X_i_ts)

        preds /= len(X)
    else:
        X, y, _, n, p = _preprocess_X_y_model(X, y, None)

        (X_tr, X_ts, y_tr, y_ts, tr_idx, ts_idx, n_tr, n_ts) = split_data(X, y, tr_idx)
        y2_ts = y2[ts_idx]

        # model.fit(X_tr, y_tr)
        model.fit(X_tr, y_tr, **kwargs)
        preds = model.predict(X_ts)

    sse = np.sum((y2_ts - preds) ** 2)
    return sse / n_ts


def _get_tr_ts_covs(
    Sigma_t,
    Sigma_s,
    tr_idx,
    ts_idx,
    alpha=None,
):
    Cov_tr_ts = Sigma_t[tr_idx, :][:, ts_idx]
    Cov_s_ts = Sigma_s[ts_idx, :][:, ts_idx]
    Cov_t_ts = Sigma_t[ts_idx, :][:, ts_idx]

    if alpha is not None:
        Cov_wp = (1 + 1. / alpha) * Sigma_t
        Cov_wp_ts = Cov_wp[ts_idx, :][:, ts_idx]
        return Cov_tr_ts, Cov_s_ts, Cov_t_ts, Cov_wp_ts

    return Cov_tr_ts, Cov_s_ts, Cov_t_ts

def _get_tr_ts_covs_corr(
    Sigma_t,
    Sigma_s,
    Cov_st,
    ts_idx,
    alpha,
    full_refit,
):

    n = Sigma_t.shape[0]
    Sigma_G = Sigma_t.copy()
    if not full_refit:
        assert(alpha is not None)
        Sigma_G *= (1 + alpha)
    Gamma = Cov_st @ np.linalg.inv(Sigma_G)
    
    IMGamma = np.eye(n) - Gamma

    Cov_N = Sigma_s - Gamma @ Sigma_G @ Gamma.T

    assert(np.allclose(Cov_N, Sigma_s - Gamma @ Cov_st.T))

    Cov_N_ts = Cov_N[ts_idx, :][:, ts_idx]

    IMGamma_ts_f = IMGamma[ts_idx, :]
    Sigma_IMG_ts_f = Sigma_t @ IMGamma_ts_f.T
    Cov_IMGY_ts = IMGamma_ts_f @ Sigma_IMG_ts_f
    if alpha is not None:
        Cov_wp = (1 + 1. / alpha) * Sigma_t
        Cov_Np = IMGamma @ Cov_wp @ IMGamma.T
        Cov_Np_ts = Cov_Np[ts_idx,:][:,ts_idx]

        return Sigma_t, Cov_N_ts, Cov_IMGY_ts, Cov_Np_ts, Gamma

    return Sigma_IMG_ts_f, Cov_N_ts, Cov_IMGY_ts, Gamma

def cp_smoother_train_test(
    model,
    X,
    y,
    tr_idx,
    Chol_t=None,
    Chol_s=None,
    Cov_st=None,
):
    X, y, _, n, p = _preprocess_X_y_model(X, y, None)

    (X_tr, X_ts, y_tr, y_ts, tr_idx, ts_idx, n_tr, n_ts) = split_data(X, y, tr_idx)

    Chol_t, Sigma_t, Chol_s, Sigma_s, _, _ = _get_covs(Chol_t, Chol_s)

    model.fit(X_tr, y_tr)
    P = model.get_linear_smoother(X, tr_idx, ts_idx)[0]
    # assert(np.allclose(P, X @ np.linalg.pinv(X)))
    # assert(n_ts == n)
    if Cov_st is None:
        # print("here")
        Cov_tr_ts, Cov_s_ts, Cov_t_ts = _get_tr_ts_covs(Sigma_t, Sigma_s, tr_idx, ts_idx)
        # assert(np.allclose(Cov_s_ts, Cov_t_ts))
        correction = (
            2 * np.diag(P @ Cov_tr_ts).mean()
            + np.diag(Cov_s_ts).mean()
            - np.diag(Cov_t_ts).mean()
        )
    else:
        tr_corr, Cov_N_ts, Cov_IMGY_ts, Gamma = _get_tr_ts_covs_corr(Sigma_t, Sigma_s, Cov_st, ts_idx, alpha=None, full_refit=True)
        correction = (
            # 2 * np.diag(tr_corr[ts_idx,:] - P @ tr_corr[tr_idx,:]).mean()
            2 * np.diag(P @ tr_corr[tr_idx,:] - Gamma[ts_idx,:] @ tr_corr).mean()
            + np.diag(Cov_N_ts).mean()
            - np.diag(Cov_IMGY_ts).mean()
        )

    return np.mean((y_ts - P @ y_tr) ** 2) + correction


def cp_general_train_test(
    model,
    X,
    y,
    tr_idx,
    Chol_t=None,
    Chol_s=None,
    Cov_st=None,
    nboot=100,
    alpha=1.0,
    use_trace_corr=True,
):
    return cp_adaptive_smoother_train_test(
        model,
        X=X,
        y=y,
        tr_idx=tr_idx,
        Chol_t=Chol_t,
        Chol_s=Chol_s,
        Cov_st=Cov_st,
        nboot=nboot,
        alpha=alpha,
        full_refit=False,
        use_trace_corr=use_trace_corr,
    )

def _get_noise_correction(Cov_s_ts, Cov_t_ts, Cov_wp_ts, use_trace_corr):
    if use_trace_corr:
        return np.diag(Cov_s_ts).mean() - np.diag(Cov_wp_ts).mean()
    else:
        return np.diag(Cov_s_ts).mean() - np.diag(Cov_t_ts).mean()
    # if use_trace_corr:
    #     return np.diag(Cov_s_ts).sum() - np.diag(Cov_wp_ts).sum()
    # else:
    #     return np.diag(Cov_s_ts).sum() - np.diag(Cov_t_ts).sum()
    
def _compute_cp_estimator(
    model,
    X, 
    X_ts, 
    y, 
    y_tr, 
    w, 
    w_tr,
    wp,
    wp_ts,
    regress_t_eps,
    tr_idx,
    ts_idx,
    Cov_st,
    Cov_tr_ts,
    full_refit,
    use_trace_corr,
    Gamma=None,
    P=None,
    P_full=None,
):
    if full_refit and not isinstance(model, LinearSelector):
        raise TypeError("model must inherit from 'LinearSelector' class")
    
    n_ts = X_ts.shape[0]

    if Gamma is not None:
        Gamma_ts = Gamma[ts_idx,:]
        IMGamma = np.eye(X.shape[0]) - Gamma
        IMGamma_ts_f = IMGamma[ts_idx,:]
    else:
        Gamma_ts = np.zeros((X_ts.shape[0], X.shape[0]))

    if isinstance(model, LinearSelector):
        if P is None:
            P = model.get_linear_smoother(X, tr_idx, ts_idx, ret_full_P=False)[0]
        if Cov_st is not None and P_full is None:
            P_full = model.get_linear_smoother(X, tr_idx, ts_idx, ret_full_P=True)[0]

    if Cov_st is None:
        regress_t_eps_ts = regress_t_eps[ts_idx]
        Np_ts = wp_ts.copy()
    else:
        regress_t_eps_ts = (IMGamma @ regress_t_eps)[ts_idx]
        Np_ts = IMGamma_ts_f @ wp

    if full_refit:
        if Cov_st is None:
            P_corr = P.copy() 
        else:
            P_corr = (IMGamma_ts_f.T @ (P_full - Gamma_ts)) 

        iter_correction = 2 * np.diag(P_corr @ Cov_tr_ts).sum() / n_ts
        corr_correction = Gamma_ts @ y
        yhat = model.predict(X, tr_idx, ts_idx, y_tr)
    else:
        iter_correction = 0
        corr_correction = Gamma_ts @ w
        yhat = model.predict(X_ts)

    in_mse = np.mean(
        (Np_ts - yhat + corr_correction) ** 2
    )

    if not use_trace_corr:
        iter_correction -= (regress_t_eps_ts**2).mean()
    
    est = in_mse + iter_correction

    return est, yhat

def cp_adaptive_smoother_train_test(
    model,
    X,
    y,
    tr_idx,
    Chol_t=None,
    Chol_s=None,
    Cov_st=None,
    nboot=100,
    alpha=1.0,
    full_refit=True,
    use_trace_corr=True,
    ret_yhats=False,
):

    X, y, _, n, _ = _preprocess_X_y_model(X, y, None)

    (X_tr, X_ts, y_tr, _, tr_idx, ts_idx, _, n_ts) = split_data(X, y, tr_idx)

    Chol_t, Sigma_t, Chol_s, Sigma_s, Chol_eps, proj_t_eps = _get_covs(
        Chol_t, Chol_s, alpha=alpha
    )

    if Cov_st is None:
        Cov_tr_ts, Cov_s_ts, Cov_t_ts, Cov_wp_ts = _get_tr_ts_covs(
            Sigma_t, Sigma_s, tr_idx, ts_idx, alpha
        )
        Gamma = None
    else:
        Cov_tr_ts, Cov_s_ts, Cov_t_ts, Cov_wp_ts, Gamma = _get_tr_ts_covs_corr(
            Sigma_t, Sigma_s, Cov_st, ts_idx, alpha, full_refit
        )

    noise_correction = _get_noise_correction(
        Cov_s_ts, 
        Cov_t_ts, 
        Cov_wp_ts,
        use_trace_corr
    )
    boot_ests = np.zeros(nboot)
    if ret_yhats:
        yhats = np.zeros((n_ts, nboot))
    for i in range(nboot):
        w, wp, eps, regress_t_eps = _blur(y, Chol_eps, proj_t_eps)
        w_tr = w[tr_idx]
        wp_ts = wp[ts_idx]

        model.fit(X_tr, w_tr)

        base_est, yhat = _compute_cp_estimator(
            model=model,
            X=X, 
            X_ts=X_ts, 
            y=y, 
            y_tr=y_tr, 
            w=w, 
            w_tr=w_tr,
            wp=wp,
            wp_ts=wp_ts,
            regress_t_eps=regress_t_eps,
            tr_idx=tr_idx,
            ts_idx=ts_idx,
            Cov_st=Cov_st,
            Cov_tr_ts=Cov_tr_ts,
            full_refit=full_refit,
            use_trace_corr=use_trace_corr,
            Gamma=Gamma,
        )
        boot_ests[i] = base_est
        if ret_yhats:
            yhats[:,i] = yhat

    cp_est = boot_ests.mean() + noise_correction
    # cp_est = boot_ests.sum() + noise_correction

    if ret_yhats:
        return cp_est, yhats

    return cp_est


def cp_bagged_train_test(
    model,
    X,
    y,
    tr_idx,
    Chol_t=None,
    Chol_s=None,
    Cov_st=None,
    full_refit=False,
    use_trace_corr=False,
    n_estimators=100,
    **kwargs,
):
    ind_est, yhats = cp_adaptive_smoother_train_test(model,
        X=X,
        y=y,
        tr_idx=tr_idx,
        Chol_t=Chol_t,
        Chol_s=Chol_s,
        Cov_st=Cov_st,
        nboot=n_estimators,
        alpha=1.,
        full_refit=full_refit,
        use_trace_corr=use_trace_corr,
        ret_yhats=True
    )

    centered_preds = yhats - yhats.mean(1)[:,None]
    return ind_est - (centered_preds**2).mean()


def cp_rf_train_test(
    model,
    X,
    y,
    tr_idx,
    Chol_t=None,
    Chol_s=None,
    Cov_st=None, ## TODO: not implemented yet...
    # n_estimators=100,
    ret_gls=False,
    full_refit=False,
    use_trace_corr=True,
    **kwargs,
):

    kwargs["chol_eps"] = Chol_t
    kwargs["idx_tr"] = tr_idx

    X, y, model, n, p = _preprocess_X_y_model(X, y, model)

    (X_tr, X_ts, y_tr, y_ts, tr_idx, ts_idx, n_tr, n_ts) = split_data(X, y, tr_idx)

    Chol_t, Sigma_t, Chol_s, Sigma_s, _, _ = _get_covs(Chol_t, Chol_s, alpha=1.0)

    model.fit(X_tr, y_tr, **kwargs)

    Ps = model.get_linear_smoother(X, tr_idx, ts_idx)#X_tr, X_ts)
    eps = model.eps_

    Cov_tr_ts, Cov_s_ts, Cov_t_ts, Cov_wp_ts = _get_tr_ts_covs(
        Sigma_t, Sigma_s, tr_idx, ts_idx, alpha=1.0
    )

    n_trees = len(Ps)

    tree_ests = np.zeros(n_trees)
    ws = np.zeros((n, n_trees))
    yhats = np.zeros((n_ts, n_trees))

    if ret_gls:
        P_gls_s = model.get_linear_smoother(
            # X_tr, X_ts, Chol=np.linalg.inv(np.linalg.cholesky(Sigma_t[tr_idx, :][:, tr_idx])).T
            X, tr_idx, ts_idx, Chol=np.linalg.inv(np.linalg.cholesky(Sigma_t[tr_idx, :][:, tr_idx])).T
        )
        tree_gls_ests = np.zeros(n_trees)
        yhats_gls = np.zeros((n_ts, n_trees))

    for i, (P_i, eps_i) in enumerate(zip(Ps, eps)):
        eps_i = eps_i.ravel()
        w = y + eps_i
        w_tr = w[tr_idx]
        regress_t_eps = eps_i
        wp = y - regress_t_eps
        wp_ts = wp[ts_idx]

        if full_refit:
            correction = 2 * np.diag(Cov_tr_ts @ P_i).sum()
        else:
            correction = 0
        if not use_trace_corr:
            correction -= (regress_t_eps[ts_idx] ** 2).sum()

        if full_refit:
            yhat = P_i @ y_tr
        else:
            yhat = P_i @ w_tr
        yhats[:, i] = yhat
        tree_ests[i] = np.sum((wp_ts - yhat) ** 2) + correction

        if ret_gls:
            P_gls_i = P_gls_s[i]
            assert(np.allclose(P_i, P_gls_i))
            if full_refit:
                gls_correction = 2 * np.diag(Cov_tr_ts @ P_gls_i).sum()
            else:
                gls_correction = 0
            if not use_trace_corr:
                gls_correction -= (regress_t_eps[ts_idx] ** 2).sum()

            if full_refit:
                yhat_gls = P_gls_i @ y_tr
            else:
                yhat_gls = P_gls_i @ w_tr
            yhats_gls[:, i] = yhat_gls
            tree_gls_ests[i] = np.sum((wp_ts - yhat_gls) ** 2) + gls_correction
            yhats_gls[:, i] = yhat_gls

    centered_preds = yhats.mean(axis=1)[:, None] - yhats
    if use_trace_corr:
        iter_indep_correction = n_trees * (
            np.diag(Cov_s_ts).sum() - np.diag(Cov_wp_ts).sum()
        )
    else:
        iter_indep_correction = n_trees * (
            np.diag(Cov_s_ts).sum() - np.diag(Cov_t_ts).sum()
        )
    ols_est = (
        tree_ests.sum() + iter_indep_correction - np.sum((centered_preds) ** 2)
    ) / (n_ts * n_trees)

    if ret_gls:
        gls_centered_preds = yhats_gls.mean(axis=1)[:, None] - yhats_gls
        return ols_est, (
            tree_gls_ests.sum()
            + iter_indep_correction
            - np.sum((gls_centered_preds) ** 2)
        ) / (n_ts * n_trees)

    return ols_est


def cb_isotropic(
    X, 
    y,
    sigma=None,
    nboot=100, 
    alpha=1.0, 
    model=LinearRegression(), 
    est_risk=True,
):

    X, y, model, n, p = _preprocess_X_y_model(X, y, model)

    if sigma is None:
        model.fit(X, y)
        pred = model.predict(X)
        sigma = np.sqrt(
            ((y - pred) ** 2).mean()
        )  # not sure how to get df for general models...

    boot_ests = np.zeros(nboot)

    for b in np.arange(nboot):
        eps = sigma * np.random.randn(n)
        w = y + eps * np.sqrt(alpha)
        wp = y - eps / np.sqrt(alpha)

        model.fit(X, w)
        yhat = model.predict(X)

        boot_ests[b] = np.sum((wp - yhat) ** 2) - np.sum(eps**2) / alpha

    return boot_ests.mean() / n + (sigma**2) * (alpha - (1 + alpha) * est_risk), model


def bag_kfoldcv(
    model,
    X,
    y,
    k=10,
    Chol_t=None,
    n_estimators=100,
):

    model = clone(model)
    X, y, model, n, p = _preprocess_X_y_model(X, y, model)

    kf = KFold(k, shuffle=True)

    bagg_model = ParametricBaggingRegressor(model, n_estimators=n_estimators)

    err = []
    kwargs = {
        "do_param_boot": False
    }  # if type(model).__name__ == 'BaggedRelaxedLasso' else dict(bootstrap_type='blur')
    for tr_idx, ts_idx in kf.split(X):
        tr_bool = np.zeros(n)
        tr_bool[tr_idx] = 1
        (X_tr, X_ts, y_tr, y_ts, tr_idx, ts_idx, n_tr, n_ts) = split_data(X, y, tr_bool)
        if Chol_t is None:
            kwargs["chol_eps"] = None
        else:
            kwargs["chol_eps"] = Chol_t[tr_idx, :][:, tr_idx]
        bagg_model.fit(X_tr, y_tr, **kwargs)
        err.append(np.mean((y_ts - bagg_model.predict(X_ts)) ** 2))

    return np.mean(err)


def bag_kmeanscv(
    model,
    X,
    y,
    coord,
    k=10,
    Chol_t=None,
    n_estimators=100,
):

    model = clone(model)
    X, y, model, n, p = _preprocess_X_y_model(X, y, model)

    groups = KMeans(n_init=10, n_clusters=k).fit(coord).labels_
    gkf = GroupKFold(k)

    bagg_model = ParametricBaggingRegressor(model, n_estimators=n_estimators)

    err = []
    kwargs = {
        "do_param_boot": False
    } # if type(model).__name__ == 'BaggedRelaxedLasso' else dict(bootstrap_type='blur')
    for tr_idx, ts_idx in gkf.split(X, groups=groups):
        tr_bool = np.zeros(n)
        tr_bool[tr_idx] = 1
        (X_tr, X_ts, y_tr, y_ts, tr_idx, ts_idx, n_tr, n_ts) = split_data(X, y, tr_bool)
        if Chol_t is None:
            kwargs["chol_eps"] = None
        else:
            kwargs["chol_eps"] = Chol_t[tr_idx, :][:, tr_idx]
        bagg_model.fit(X_tr, y_tr, **kwargs)
        err.append(np.mean((y_ts - bagg_model.predict(X_ts)) ** 2))

    return np.mean(err)


def simple_train_test_split(model, X, y, tr_idx, **kwargs):

    model = clone(model)

    X_tr, X_ts, y_tr, y_ts, tr_idx, ts_idx, n_tr, n_ts = split_data(X, y, tr_idx)
    
    model.fit(X_tr, y_tr, **kwargs)
    preds = model.predict(X_ts)

    return np.mean((y_ts - preds)**2)


def kfoldcv(model, X, y, k=10, **kwargs):

    model = clone(model)

    kfcv_res = cross_validate(
        model,
        X,
        y,
        scoring="neg_mean_squared_error",
        cv=KFold(k, shuffle=True),
        error_score="raise",
        fit_params=kwargs,
    )
    return -np.mean(kfcv_res["test_score"])  # , model


def my_timeseriessplit(X, k=10, min_train_size=0, max_train_size=None, gap=0, test_size=None):
    ts_splitter = TimeSeriesSplit(n_splits=k, max_train_size=max_train_size, gap=gap, test_size=test_size)
    for tr, ts in ts_splitter.split(X):
        if tr.sum() > min_train_size:
            yield tr, ts

def timeseriescv(model, X, y, k=10, min_train_size=0, max_train_size=None, gap=0, test_size=None, **kwargs):

    model = clone(model)

    tscv_res = cross_validate(
        model,
        X,
        y,
        scoring="neg_mean_squared_error",
        cv=my_timeseriessplit(X, k=k, min_train_size=min_train_size, max_train_size=max_train_size, gap=gap, test_size=test_size),
        error_score="raise",
        fit_params=kwargs,
    )
    return -np.mean(tscv_res["test_score"])  # , model


def kmeanscv(model, X, y, coord, k=10, **kwargs):

    groups = KMeans(n_init=10, n_clusters=k).fit(coord).labels_
    spcv_res = cross_validate(
        model,
        X,
        y,
        scoring="neg_mean_squared_error",
        cv=GroupKFold(k),
        groups=groups,
        fit_params=kwargs,
    )

    return -np.mean(spcv_res["test_score"])  # , model


def test_set_estimator(
    model,
    X,
    y,
    y_test,
    Chol_t=None,
    Chol_eps=None,
    Theta_p=None,
    Theta_e=None,
    est_risk=True,
):

    model = clone(model)

    multiple_X = isinstance(X, list)

    if multiple_X:
        n = X[0].shape[0]

    else:
        n = X.shape[0]

    (
        Chol_t,
        Sigma_t,
        Chol_eps,
        Sigma_eps,
        Prec_eps,
        proj_t_eps,
        Theta_p,
        Chol_p,
        Sigma_t_Theta_p,
        Aperp,
    ) = _compute_matrices(n, Chol_t, Chol_eps, Theta_p)

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
    sse = np.sum((Chol_p @ (y_test - preds)) ** 2)
    return (sse - np.diag(Sigma_t_Theta_p).sum() * est_risk) / n, model
