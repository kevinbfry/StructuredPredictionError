import numpy as np
from itertools import product
from sklearn.linear_model import LinearRegression
# from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate, GroupKFold, KFold, TimeSeriesSplit
from sklearn.cluster import KMeans

from sklearn.base import clone
from sklearn.utils.validation import check_X_y

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

    # model.fit(X_tr, y_tr, **kwargs)
    # preds = model.predict(X_ts)
    # preds = model.predict(X_ts, **kwargs)

    if bagg:
        base_model = clone(model)
        model = ParametricBaggingRegressor(base_model, n_estimators=100)

    if alpha is not None:
        w, eps = _blur(y, np.sqrt(alpha) * Chol_t)
        w_tr = w[tr_idx]
        model.fit(X_tr, w_tr, **kwargs)
    else:
        # if model.__class__.__name__ == "BlurredForest":
        if isinstance(model, (ParametricBaggingRegressor, BlurredForest)):
            if "chol_eps" in kwargs:
                pass
                # kwargs["chol_eps"] = kwargs["chol_eps"]#[tr_idx, :][:, tr_idx]
                # kwargs["idx_tr"] = tr_idx
            else:
                kwargs["chol_eps"] = Chol_t#[tr_idx, :][:, tr_idx]
            kwargs["idx_tr"] = tr_idx
        model.fit(X_tr, y_tr, **kwargs)

    # print(model.__class__.__name__, full_refit, gls)
    # print(full_refit is None, full_refit, full_refit is None or full_refit)

    if full_refit is None or full_refit:
        # print(model.__class__.__name__, full_refit, gls)
        ## TODO: change to isinstance(LinearSelector)
        # if model.__class__.__name__ in ["Tree", "RelaxedLasso"]:
        if isinstance(model, (Tree, RelaxedLasso)):
            P = model.get_linear_smoother(X, tr_idx, ts_idx)#X_tr, X_ts)
            preds = P @ y_tr
        elif isinstance(model, BlurredForest):
            # if gls is None or not gls:
            #     # print("gls None", gls)
            #     preds = model.predict(X_ts, full_refit=full_refit)
            # else:
            #     # print("gls not None", gls)
            #     Chol_t_inv_tr = np.linalg.inv(np.linalg.cholesky(
            #         (Chol_t @ Chol_t.T)[tr_idx,:][:,tr_idx]
            #     )).T
            #     # print("gls", gls)
            #     # print("Chol", Chol_t_tr)
            #     # print("Sigma", Chol_t_tr @ Chol_t_tr.T)
            #     preds = model.predict(
            #         X_ts, 
            #         full_refit=full_refit, 
            #         Chol=Chol_t_inv_tr
            #     )
        
            chol = None if not gls else np.linalg.inv(np.linalg.cholesky(
                                            (Chol_t @ Chol_t.T)[tr_idx,:][:,tr_idx]
                                        )).T
            preds = model.predict(X, tr_idx, ts_idx, full_refit=full_refit, Chol=chol)
        elif isinstance(model, ParametricBaggingRegressor):
            preds = model.predict(X, tr_idx, ts_idx, full_refit=full_refit)
        else:
            preds = model.predict(X_ts)

    else:
        ## TODO: all the conditional stuff I have for full refit
        # if model.__class__.__name__ == "BlurredForest":
        if isinstance(model, BlurredForest):
            if gls is None or not gls:
                # print("gls None", gls)
                preds = model.predict(X_ts, full_refit=full_refit)
                # preds = model.predict(X, tr_idx, ts_idx, full_refit=full_refit)
            else:
                # print("gls not None", gls)
                Chol_t_inv_tr = np.linalg.inv(np.linalg.cholesky(
                    (Chol_t @ Chol_t.T)[tr_idx,:][:,tr_idx]
                )).T
                # print("gls", gls)
                # print("Chol", Chol_t_tr)
                # print("Sigma", Chol_t_tr @ Chol_t_tr.T)
                # preds = model.predict(
                #     X_ts, 
                #     full_refit=full_refit, 
                #     Chol=Chol_t_inv_tr
                # )
                preds = model.predict(
                    X,
                    tr_idx,
                    ts_idx, 
                    full_refit=full_refit, 
                    Chol=Chol_t_inv_tr
                )
        elif isinstance(model, ParametricBaggingRegressor):
            preds = model.predict(
                    X,
                    tr_idx,
                    ts_idx, 
                    full_refit=full_refit, 
                )
        else:
            # print("using model predict function")
            preds = model.predict(X_ts)


    # sse = np.sum((y2_ts - preds) ** 2)
    # return sse / n_ts
    return np.mean((y2_ts - preds)**2)


# ## \| Y^b_1 - g(Y^b_2) \|_2^2
# def para_boot_est(

# )


## \| Y - g(Y) \|_2^2 + 1/(B-1)\sum_i (Y^b_i - \bar Y_B)g(Y^b_i)
def efron_boot_est(
    model,
    X,
    y,
    nboot=100,
    Chol_t=None,
    alpha=None,
    Chol_s=None,
    tr_idx=None,
):
    X, y, model, n, p = _preprocess_X_y_model(X, y, model)

    boot_samples = np.zeros((n, nboot))
    boot_preds = np.zeros((n, nboot))
    for b in np.arange(nboot):
        yb = y + np.sqrt(alpha)*Chol_t @ np.random.randn(n)
        boot_samples[:,b] = yb

        model.fit(X, yb)
        preds = model.predict(X)
        boot_preds[:,b] = preds

    model.fit(X,y)
    obs_preds = model.predict(X)

    boot_corr = 2*np.sum((boot_samples - boot_samples.mean(1)[:, None]).T @ boot_preds) / ((nboot - 1)*alpha)

    sigmasq = alpha*Chol_t[0,0]**2
    return np.mean((y - obs_preds)**2) + boot_corr/n #+ sigmasq



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
    Sigma_G = Sigma_t
    if not full_refit:
        assert(alpha is not None)
        Sigma_G *= (1 + alpha)
    Gamma = Cov_st @ np.linalg.inv(Sigma_G)

    Gamma_ts = Gamma[ts_idx,:]
    IMGamma = np.eye(n) - Gamma

    Cov_N = Sigma_s - Gamma @ Sigma_G @ Gamma.T

    assert(np.allclose(Cov_N, Sigma_s - Gamma @ Cov_st.T))

    Cov_N_ts = Cov_N[ts_idx, :][:, ts_idx]

    IMGamma_ts_f = IMGamma[ts_idx, :]
    Sigma_IMG_ts_f = Sigma_t @ IMGamma_ts_f.T
    Cov_IMGY_ts = IMGamma_ts_f @ Sigma_IMG_ts_f
    if alpha is not None:
        Cov_wp = (1 + 1 / alpha) * Sigma_t
        Cov_Np = IMGamma @ Cov_wp @ IMGamma.T
        Cov_Np_ts = Cov_Np[ts_idx,:][:,ts_idx]

        return Sigma_t, Cov_N_ts, Cov_IMGY_ts, Cov_Np_ts, Gamma_ts, IMGamma, IMGamma_ts_f

    return Sigma_IMG_ts_f, Cov_N_ts, Cov_IMGY_ts

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

    # P = X_ts @ np.linalg.inv(X_tr.T @ X_tr) @ X_tr.T
    model.fit(X_tr, y_tr)
    P = model.get_linear_smoother(X, tr_idx, ts_idx)

    if Cov_st is None:
        Cov_tr_ts, Cov_s_ts, Cov_t_ts = _get_tr_ts_covs(Sigma_t, Sigma_s, tr_idx, ts_idx)
        correction = (
            2 * np.diag(Cov_tr_ts @ P).mean()
            + np.diag(Cov_s_ts).mean()
            - np.diag(Cov_t_ts).mean()
        )
    else:
        tr_corr, Cov_N_ts, Cov_IMGY_ts = _get_tr_ts_covs_corr(Sigma_t, Sigma_s, Cov_st, ts_idx, alpha=None, full_refit=True)
        correction = (
            2 * np.diag(tr_corr[ts_idx,:] - P @ tr_corr[tr_idx,:]).mean()
            + np.diag(Cov_N_ts).mean()
            - np.diag(Cov_IMGY_ts).mean()
        )

    return np.mean((y_ts - P @ y_tr) ** 2) + correction


def cp_linear_train_test(
    model,
    X,
    y,
    tr_idx,
    Chol_t=None,
    Chol_s=None,
):
    if model.__class__.__name__ == "LinearRegression":
        return cp_smoother_train_test(
            model,
            X,
            y,
            tr_idx,
            Chol_t,
            Chol_s,
        )
    else:
        raise ValueError(
            "'cp_linear_train_test' intended only for "
            "LinearRegression model, and 'model' arg "
            "exists only for back-compatibility. "
            "Please use 'cp_smoother_train_test' if you "
            "wish to use a different linear smoothing model."
        )
        # return cp_smoother_train_test(
        #     LinearRegression(fit_intercept=False),
        #     X,
        #     y,
        #     tr_idx,
        #     Chol_t,
        #     Chol_s,
        # )

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
    
    # X, y, _, n, _ = _preprocess_X_y_model(X, y, None)

    # (X_tr, X_ts, _, _, tr_idx, ts_idx, _, n_ts) = split_data(X, y, tr_idx)

    # Chol_t, Sigma_t, Chol_s, Sigma_s, Chol_eps, proj_t_eps = _get_covs(
    #     Chol_t, Chol_s, alpha=alpha
    # )

    # if Cov_st is None:
    #     _, Cov_s_ts, Cov_t_ts, Cov_wp_ts = _get_tr_ts_covs(
    #         Sigma_t, Sigma_s, tr_idx, ts_idx, alpha
    #     )
    #     Gamma_ts = np.zeros((n_ts, n))
    # else:
    #     _, Cov_s_ts, Cov_t_ts, Cov_wp_ts, Gamma_ts, IMGamma, IMGamma_ts_f = _get_tr_ts_covs_corr(
    #         Sigma_t, Sigma_s, Cov_st, ts_idx, alpha, full_refit=False
    #     )

    # if use_trace_corr:
    #     noise_correction = np.diag(Cov_s_ts).mean() - np.diag(Cov_wp_ts).mean()
    # else:
    #     noise_correction = np.diag(Cov_s_ts).mean() - np.diag(Cov_t_ts).mean()
    # boot_ests = np.zeros(nboot)
    # for i in range(nboot):
    #     w, wp, _, regress_t_eps = _blur(y, Chol_eps, proj_t_eps)
    #     w_tr = w[tr_idx]
    #     wp_ts = wp[ts_idx]

    #     model.fit(X_tr, w_tr)

    #     if Cov_st is None:
    #         regress_t_eps = regress_t_eps[ts_idx]
    #         Np_ts = wp_ts
    #     else:
    #         regress_t_eps = IMGamma.T[ts_idx,:] @ regress_t_eps
    #         Np_ts = IMGamma_ts_f @ wp

    #     boot_ests[i] = np.mean(
    #         (Np_ts - model.predict(X_ts) + Gamma_ts @ w) ** 2
    #     )

    #     if not use_trace_corr:
    #         boot_ests[i] -= (regress_t_eps**2).mean()

    # return boot_ests.mean() + noise_correction

def _get_noise_correction(Cov_s_ts, Cov_t_ts, Cov_wp_ts, use_trace_corr):
    if use_trace_corr:
        return np.diag(Cov_s_ts).mean() - np.diag(Cov_wp_ts).mean()
    else:
        return np.diag(Cov_s_ts).mean() - np.diag(Cov_t_ts).mean()
    
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
    
    if Gamma is not None:
        Gamma_ts = Gamma[ts_idx,:]
        IMGamma = np.eye(X.shape[0]) - Gamma
        IMGamma_ts_f = IMGamma[ts_idx,:]
    else:
        Gamma_ts = np.zeros((X_ts.shape[0], X.shape[0]))

    if isinstance(model, LinearSelector):
        if P is None:
            P = model.get_linear_smoother(X, tr_idx, ts_idx)
        if Cov_st is not None and P_full is None:
            P_full = model.get_linear_smoother(X, tr_idx, ts_idx, ret_full_P=True)

    if Cov_st is None:
        regress_t_eps = regress_t_eps[ts_idx]
        Np_ts = wp_ts
    else:
        regress_t_eps = IMGamma.T[ts_idx,:] @ regress_t_eps
        Np_ts = IMGamma_ts_f @ wp

    if full_refit:
        # P = model.get_linear_smoother(X, tr_idx, ts_idx)#X_tr, X_ts)
        if Cov_st is None:
            P_corr = P 
        else:
            P_corr = (IMGamma_ts_f.T @ (P_full - Gamma_ts)) 

        iter_correction = 2 * np.diag(Cov_tr_ts @ P_corr).mean()
        yhat = P @ y_tr
        assert(np.allclose(P @ w_tr, model.predict(X_ts)))
        in_mse = np.mean(
            (Np_ts - yhat + Gamma_ts @ y)**2
        )
    else:
        iter_correction = 0
        yhat = model.predict(X_ts) #if P is None else P @ w_tr
        # yhat = P @ w_tr
        in_mse = np.mean(
            (Np_ts - yhat + Gamma_ts @ w) ** 2
        )

    if not use_trace_corr:
        iter_correction -= (regress_t_eps**2).mean()
    
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
        # Gamma = None
        Gamma_ts = np.zeros((n_ts, n))
    else:
        Cov_tr_ts, Cov_s_ts, Cov_t_ts, Cov_wp_ts, Gamma_ts, IMGamma, IMGamma_ts_f = _get_tr_ts_covs_corr(
        # Cov_tr_ts, Cov_s_ts, Cov_t_ts, Cov_wp_ts, Gamma = _get_tr_ts_covs_corr(
            Sigma_t, Sigma_s, Cov_st, ts_idx, alpha, full_refit
        )
        IMGamma = np.eye(n) - Gamma
        Gamma_ts = Gamma[ts_idx,:]
        IMGamma_ts_f = IMGamma[ts_idx,:]

    if use_trace_corr:
        noise_correction = np.diag(Cov_s_ts).mean() - np.diag(Cov_wp_ts).mean()
    else:
        noise_correction = np.diag(Cov_s_ts).mean() - np.diag(Cov_t_ts).mean()
    # noise_correction = _get_noise_correction(
    #     Cov_s_ts, 
    #     Cov_t_ts, 
    #     Cov_wp_ts,
    #     use_trace_corr
    # )
    boot_ests = np.zeros(nboot)
    for i in range(nboot):
        w, wp, eps, regress_t_eps = _blur(y, Chol_eps, proj_t_eps)
        w_tr = w[tr_idx]
        wp_ts = wp[ts_idx]

        model.fit(X_tr, w_tr)

        if Cov_st is None:
            regress_t_eps = regress_t_eps[ts_idx]
            Np_ts = wp_ts
        else:
            regress_t_eps = IMGamma.T[ts_idx,:] @ regress_t_eps
            Np_ts = IMGamma_ts_f @ wp

        if full_refit:
            P = model.get_linear_smoother(X, tr_idx, ts_idx)#X_tr, X_ts)
            if Cov_st is None:
                P_corr = P 
            else:
                P_full = model.get_linear_smoother(X, tr_idx, ts_idx, ret_full_P=True)
                P_corr = (IMGamma_ts_f.T @ (P_full - Gamma_ts)) 

            iter_correction = 2 * np.diag(Cov_tr_ts @ P_corr).mean()
            boot_ests[i] = np.mean(
                (Np_ts - P @ y_tr + Gamma_ts @ y)**2
            )
        else:
            iter_correction = 0
            boot_ests[i] = np.mean(
                (Np_ts - model.predict(X_ts) + Gamma_ts @ w) ** 2
            )

        if not use_trace_corr:
            iter_correction -= (regress_t_eps**2).mean()

        boot_ests[i] += iter_correction
        # base_est, _ = _compute_cp_estimator(
        #     model=model,
        #     X=X, 
        #     X_ts=X_ts, 
        #     y=y, 
        #     y_tr=y_tr, 
        #     w=w, 
        #     w_tr=w_tr,
        #     wp=wp,
        #     wp_ts=wp_ts,
        #     regress_t_eps=regress_t_eps,
        #     tr_idx=tr_idx,
        #     ts_idx=ts_idx,
        #     Cov_st=Cov_st,
        #     Cov_tr_ts=Cov_tr_ts,
        #     full_refit=full_refit,
        #     use_trace_corr=use_trace_corr,
        #     Gamma=Gamma,
        # )
        # boot_ests[i] = base_est

    return boot_ests.mean() + noise_correction

# def cp_bagged_train_test(
#     model,
#     X,
#     y,
#     tr_idx,
#     Chol_t=None,
#     Chol_s=None,
#     Cov_st=None,
#     full_refit=False,
#     use_trace_corr=False,
#     n_estimators=100,
#     **kwargs,
# ):
#     if full_refit:
#         assert(isinstance(model, LinearSelector))

#     # model = clone(model)
#     kwargs["chol_eps"] = Chol_t
#     kwargs["idx_tr"] = tr_idx
#     kwargs["do_param_boot"] = True

#     # X, y, _, n, p = _preprocess_X_y_model(X, y, None)
#     X, y, model, n, p = _preprocess_X_y_model(X, y, model)

#     (X_tr, X_ts, y_tr, y_ts, tr_idx, ts_idx, n_tr, n_ts) = split_data(X, y, tr_idx)

#     Chol_t, Sigma_t, Chol_s, Sigma_s, _, _ = _get_covs(Chol_t, Chol_s, alpha=1.0)

#     # model.fit(X_tr, y_tr, **kwargs)
#     bagg_model = ParametricBaggingRegressor(model, n_estimators=n_estimators)
#     bagg_model.fit(X_tr, y_tr, **kwargs)

#     Ps = bagg_model.get_linear_smoother(X, tr_idx, ts_idx)#X_tr, X_ts)
#     eps = bagg_model.eps_

#     Cov_tr_ts, Cov_s_ts, Cov_t_ts, Cov_wp_ts = _get_tr_ts_covs(
#         Sigma_t, Sigma_s, tr_idx, ts_idx, alpha=1.0
#     )

#     n_ests = len(Ps)

#     base_ests = np.zeros(n_ests)
#     ws = np.zeros((n, n_ests))
#     yhats = np.zeros((n_ts, n_ests))

#     for i, (P_i, eps_i) in enumerate(zip(Ps, eps)):
#         eps_i = eps_i.ravel()
#         w = y + eps_i
#         w_tr = w[tr_idx]
#         regress_t_eps = eps_i
#         wp = y - regress_t_eps
#         wp_ts = wp[ts_idx]

#         if full_refit:
#             correction = 2 * np.diag(Cov_tr_ts @ P_i).sum()
#         else:
#             correction = 0
#         if not use_trace_corr:
#             correction -= (regress_t_eps[ts_idx] ** 2).sum()

#         if full_refit:
#             yhat = P_i @ y_tr
#         else:
#             yhat = P_i @ w_tr
#         yhats[:, i] = yhat
#         base_ests[i] = np.sum((wp_ts - yhat) ** 2) + correction

#     centered_preds = yhats.mean(axis=1)[:, None] - yhats
#     if use_trace_corr:
#         iter_indep_correction = n_ests * (
#             np.diag(Cov_s_ts).sum() - np.diag(Cov_wp_ts).sum()
#         )
#     else:
#         iter_indep_correction = n_ests * (
#             np.diag(Cov_s_ts).sum() - np.diag(Cov_t_ts).sum()
#         )
#     est = (
#         base_ests.sum() + iter_indep_correction - np.sum((centered_preds) ** 2)
#     ) / (n_ts * n_ests)

#     return est

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
    if full_refit:
        assert(isinstance(model, LinearSelector))

    # model = clone(model)
    kwargs["chol_eps"] = Chol_t
    kwargs["idx_tr"] = tr_idx
    kwargs["do_param_boot"] = True

    # X, y, _, n, p = _preprocess_X_y_model(X, y, None)
    X, y, model, n, p = _preprocess_X_y_model(X, y, model)

    (X_tr, X_ts, y_tr, y_ts, tr_idx, ts_idx, n_tr, n_ts) = split_data(X, y, tr_idx)

    Chol_t, Sigma_t, Chol_s, Sigma_s, _, _ = _get_covs(Chol_t, Chol_s, alpha=1.0)

    # model.fit(X_tr, y_tr, **kwargs)
    bagg_model = ParametricBaggingRegressor(model, n_estimators=n_estimators)
    bagg_model.fit(X_tr, y_tr, **kwargs)

    if full_refit:
        Ps = bagg_model.get_linear_smoother(X, tr_idx, ts_idx, ret_full_P=False)#X_tr, X_ts)
    else:
        Ps = None
    eps = bagg_model.eps_

    if Cov_st is None:
        Cov_tr_ts, Cov_s_ts, Cov_t_ts, Cov_wp_ts = _get_tr_ts_covs(
            Sigma_t, Sigma_s, tr_idx, ts_idx, 1.0
        )
        # Gamma_ts = np.zeros((n_ts, n))
        Gamma = None
    else:
        # Cov_tr_ts, Cov_s_ts, Cov_t_ts, Cov_wp_ts, Gamma_ts, IMGamma, IMGamma_ts_f = _get_tr_ts_covs_corr(
        Cov_tr_ts, Cov_s_ts, Cov_t_ts, Cov_wp_ts, Gamma = _get_tr_ts_covs_corr(
            Sigma_t, Sigma_s, Cov_st, ts_idx, 1.0, full_refit
        )
        
    if Cov_st is not None and full_refit:
        P_fulls = bagg_model.get_linear_smoother(X, tr_idx, ts_idx, ret_full_P=True)
    else:
        P_fulls = None

    n_ests = len(eps)

    base_ests = np.zeros(n_ests)
    # ws = np.zeros((n, n_ests))
    yhats = np.zeros((n_ts, n_ests))

    noise_correction = _get_noise_correction(
        Cov_s_ts,
        Cov_t_ts, 
        Cov_wp_ts, 
        use_trace_corr
    )

    # for i, (P_i, eps_i) in enumerate(zip(Ps, eps)):
    for i, eps_i in enumerate(eps):
        eps_i = eps_i#.ravel()
        w = y + eps_i
        w_tr = w[tr_idx]
        regress_t_eps = eps_i
        wp = y - regress_t_eps
        wp_ts = wp[ts_idx]

        P_i = Ps[i] if Ps is not None else None
        P_full_i = P_fulls[i] if P_fulls is not None else None
        model_i = bagg_model.estimators_[i]

        # if Cov_st is None:
        #     regress_t_eps = regress_t_eps[ts_idx]
        #     Np_ts = wp_ts
        # else:
        #     regress_t_eps = IMGamma.T[ts_idx,:] @ regress_t_eps
        #     Np_ts = IMGamma_ts_f @ wp

        # if full_refit:
        #     if Cov_st is None:
        #         P_corr = P_i
        #     else:
        #         P_full_i = P_fulls[i]
        #         P_corr = (IMGamma_ts_f.T @ (P_full_i - Gamma_ts)) 

        #     iter_correction = 2 * np.diag(Cov_tr_ts @ P_corr).sum()
        #     base_ests[i] = np.sum(
        #         (Np_ts - P_i @ y_tr + Gamma_ts @ y)**2
        #     )
        #     yhat = P_i @ y_tr
        # else:
        #     iter_correction = 0
        #     base_ests[i] = np.sum(
        #         (Np_ts - P_i @ w_tr + Gamma_ts @ w) ** 2
        #     )
        #     yhat = P_i @ w_tr

        # if not use_trace_corr:
        #     iter_correction -= (regress_t_eps**2).sum()

        # base_ests[i] += iter_correction

        base_est, yhat = _compute_cp_estimator(
            model_i,
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
            Gamma=Gamma,
            P=P_i,
            P_full=P_full_i,
        )
        base_ests[i] = base_est
        yhats[:,i] = yhat

    # if use_trace_corr:
    #     noise_correction = n_ests * (
    #         np.diag(Cov_s_ts).sum() - np.diag(Cov_wp_ts).sum()
    #     )
    # else:
    #     noise_correction = n_ests * (
    #         np.diag(Cov_s_ts).sum() - np.diag(Cov_t_ts).sum()
    #     )

    centered_preds = yhats.mean(axis=1)[:, None] - yhats
    # assert(np.all(centered_preds == 0))
    # est = (
    #     base_ests.sum() - np.sum((centered_preds) ** 2)
    # ) / (n_ests * n_ts) + noise_correction
    # est = (
    #      - np.sum((centered_preds) ** 2)
    # ) / (n_ests * n_ts) + base_ests.mean() + noise_correction
    est = (
        base_ests.mean() - np.mean((centered_preds) ** 2) + noise_correction
    )

    return est


def cp_rf_train_test(
    model,
    X,
    y,
    tr_idx,
    Chol_t=None,
    Chol_s=None,
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


# def cb(
#     X,
#     y,
#     Chol_t=None,
#     Chol_eps=None,
#     Theta_p=None,
#     nboot=100,
#     model=LinearRegression(),
#     est_risk=True,
# ):

#     X, y, model, n, p = _preprocess_X_y_model(X, y, model)

#     (
#         Chol_t,
#         Sigma_t,
#         Chol_eps,
#         Sigma_eps,
#         Prec_eps,
#         proj_t_eps,
#         Theta_p,
#         Chol_p,
#         Sigma_t_Theta_p,
#         Aperp,
#     ) = _compute_matrices(n, Chol_t, Chol_eps, Theta_p)

#     Sigma_eps_Theta_p = Sigma_eps @ Theta_p

#     boot_ests = np.zeros(nboot)

#     for b in np.arange(nboot):
#         w, wp, eps, regress_t_eps = _blur(y, Chol_eps, proj_t_eps)

#         model.fit(X, w)
#         yhat = model.predict(X)

#         # boot_ests[b] = np.sum((wp - yhat)**2) - (regress_t_eps.T.dot(Theta_p @ regress_t_eps)).sum()
#         boot_ests[b] = np.sum((Chol_p @ (wp - yhat)) ** 2) - np.sum(
#             (Chol_p @ regress_t_eps) ** 2
#         )

#     return (
#         boot_ests.mean()
#         - np.diag(Sigma_t_Theta_p).sum() * est_risk
#         + np.diag(Sigma_eps_Theta_p).sum() * (1 - est_risk)
#     ) / n, model


# def blur_linear(
#     X,
#     y,
#     Chol_t=None,
#     Chol_eps=None,
#     Theta_p=None,
#     Theta_e=None,
#     alpha=None,
#     nboot=100,
#     model=LinearRegression(),
#     est_risk=True,
# ):

#     X, y, model, n, p = _preprocess_X_y_model(X, y, model)

#     (
#         Chol_t,
#         Sigma_t,
#         Chol_eps,
#         Sigma_eps,
#         Prec_eps,
#         proj_t_eps,
#         Theta_p,
#         Chol_p,
#         Sigma_t_Theta_p,
#         Aperp,
#     ) = _compute_matrices(n, Chol_t, Chol_eps, Theta_p)

#     # Theta_e = Prec_eps
#     # Chol_e = np.linalg.cholesky(Theta_e)
#     if Theta_e is not None:
#         Chol_e = np.linalg.cholesky(Theta_e)
#     else:
#         Theta_e = Prec_eps
#         Chol_e = np.linalg.cholesky(Theta_e)
#         # Theta_e = np.eye(n)
#         # Chol_e = np.eye(n)
#     X_e = Chol_e.T @ X

#     # P = X @ np.linalg.inv(X.T @ X) @ X.T
#     P = X @ np.linalg.inv(X_e.T @ X_e) @ X.T @ Theta_e

#     boot_ests = np.zeros(nboot)

#     # assert(np.allclose(Sigma_t_Theta_p, np.eye(n)))
#     # assert(np.allclose(np.linalg.inv(Theta_p),Sigma_t))
#     # assert(np.allclose(proj_t_eps, np.eye(n) / alpha))
#     # assert(np.allclose(Sigma_eps @ Theta_p, np.eye(n) * alpha))
#     # assert(np.allclose(proj_t_eps @ Sigma_t_Theta_p, np.eye(n) / alpha))
#     # assert(np.allclose(P @ Sigma_eps @ P.T @ Theta_p, Sigma_eps @ Theta_p))
#     # assert(np.allclose(P @ Sigma_eps @ P.T @ Theta_p, np.eye(n)*alpha))

#     for b in np.arange(nboot):
#         w, wp, eps, regress_t_eps = _blur(y, Chol_eps, proj_t_eps)

#         model.fit(X_e, Chol_e.T @ w)
#         # model.fit(X, w)
#         yhat = model.predict(X)
#         # yhat2 = P @ w
#         # assert(np.allclose(yhat, yhat2))

#         # boot_ests[b] = np.sum((wp - yhat)**2) - np.sum(regress_t_eps**2) - np.sum((P @ eps)**2)
#         # boot_ests[b] = np.sum((wp - yhat)**2) \
#         # 				- np.diag(proj_t_eps @ Sigma_t).sum() \
#         # 				- np.diag(P @ Sigma_eps @ P.T).sum()

#         # boot_ests[b] = np.sum((Chol_p@(wp - yhat))**2) - np.sum((Chol_p@regress_t_eps)**2) - np.sum((Chol_p @ P @ eps)**2)
#         # boot_ests[b] = np.sum((wp - yhat).T.dot(Theta_p.dot((wp - yhat)))) \
#         # 				- np.sum(regress_t_eps.T.dot(Theta_p.dot(regress_t_eps))) \
#         # 				- np.sum((P @ eps).T.dot(Theta_p.dot(P @ eps)))

#         boot_ests[b] = np.sum((wp - yhat).T.dot(Theta_p.dot((wp - yhat))))

#     # print(np.diag(Sigma_t_Theta_p).sum())
#     return (
#         boot_ests.mean()
#         - np.diag(proj_t_eps @ Sigma_t_Theta_p).sum()
#         - np.diag(P @ Sigma_eps @ P.T @ Theta_p).sum()
#         - np.diag(Sigma_t_Theta_p).sum() * est_risk
#     ) / n, model
#     # return (boot_ests.mean() - np.diag(Sigma_t).sum()*est_risk) / n, model


# def _compute_correction(
#     y,
#     w,
#     wp,
#     P,
#     Aperp,
#     regress_t_eps,
#     Theta_p,
#     Chol_p,
#     Sigma_t_Theta_p,
#     proj_t_eps,
#     full_rand,
#     use_expectation,
#     est_risk,
# ):
#     yhat = P @ y if full_rand else P @ w
#     PAperp = P @ Aperp
#     Theta_p_PAperp = Theta_p @ PAperp
#     # print(Sigma_t_Theta_p)
#     # print(proj_t_eps)

#     if use_expectation:
#         # boot_est = np.sum((wp - yhat)**2)
#         boot_est = np.sum((Chol_p @ (wp - yhat)) ** 2)
#     else:
#         boot_est = np.sum((Chol_p @ (wp - yhat)) ** 2) - np.sum(
#             (Chol_p @ regress_t_eps) ** 2
#         )
#         if full_rand:
#             boot_est += 2 * regress_t_eps.T.dot(Theta_p_PAperp.dot(regress_t_eps))

#     expectation_correction = 0.0
#     if full_rand:
#         expectation_correction += 2 * np.diag(Sigma_t_Theta_p @ PAperp).sum()
#     if use_expectation:
#         t_epsinv_t = proj_t_eps @ Sigma_t_Theta_p
#         # print(t_epsinv_t)
#         expectation_correction -= np.diag(t_epsinv_t).sum()
#         if full_rand:
#             expectation_correction += 2 * np.diag(t_epsinv_t @ PAperp).sum()

#     return (
#         boot_est + expectation_correction - np.diag(Sigma_t_Theta_p).sum() * est_risk,
#         yhat,
#     )


# def blur_linear_selector(
#     X,
#     y,
#     Chol_t=None,
#     Chol_eps=None,
#     Theta_p=None,
#     Theta_e=None,
#     model=RelaxedLasso(),
#     rand_type="full",
#     use_expectation=False,
#     est_risk=True,
# ):

#     X, y, model, n, p = _preprocess_X_y_model(X, y, model)

#     full_rand = _get_rand_bool(rand_type)

#     (
#         Chol_t,
#         Sigma_t,
#         Chol_eps,
#         Sigma_eps,
#         Prec_eps,
#         proj_t_eps,
#         Theta_p,
#         Chol_p,
#         Sigma_t_Theta_p,
#         Aperp,
#     ) = _compute_matrices(n, Chol_t, Chol_eps, Theta_p)

#     w, wp, eps, regress_t_eps = _blur(y, Chol_eps, proj_t_eps)

#     if Theta_e is not None:
#         Chol_e = np.linalg.cholesky(Theta_e)
#     else:
#         # Theta_e = Prec_eps
#         # Chol_e = np.linalg.cholesky(Theta_e)
#         Chol_e = np.eye(n)
#     X_e = Chol_e.T @ X

#     # model.fit(X, w)
#     model.fit(X_e, Chol_e.T @ w)

#     P = model.get_linear_smoother(X)

#     est, _ = _compute_correction(
#         y,
#         w,
#         wp,
#         P,
#         Aperp,
#         regress_t_eps,
#         Theta_p,
#         Chol_p,
#         Sigma_t_Theta_p,
#         proj_t_eps,
#         full_rand,
#         use_expectation,
#         est_risk,
#     )

#     return est / n, model, w


# def get_estimate_terms(
#     y,
#     P,
#     eps,
#     Sigma_t,
#     Theta_p,
#     Chol_p,
#     Sigma_t_Theta_p,
#     proj_t_eps,
#     Aperp,
#     full_rand,
#     use_expectation,
#     est_risk,
# ):
#     n_est = len(P)
#     n = y.shape[0]
#     tree_ests = np.zeros(n_est)
#     ws = np.zeros((n, n_est))
#     yhats = np.zeros((n, n_est))
#     for i, (P_i, eps_i) in enumerate(zip(P, eps)):
#         eps_i = eps_i.ravel()
#         w = y + eps_i
#         regress_t_eps = proj_t_eps @ eps_i
#         wp = y - regress_t_eps

#         tree_ests[i], yhat = _compute_correction(
#             y,
#             w,
#             wp,
#             P_i,
#             Aperp,
#             regress_t_eps,
#             Theta_p,
#             Chol_p,
#             Sigma_t_Theta_p,
#             proj_t_eps,
#             full_rand,
#             use_expectation,
#             est_risk,
#         )

#         ws[:, i] = w
#         yhats[:, i] = yhat

#     centered_preds = yhats.mean(axis=1)[:, None] - yhats

#     return tree_ests, centered_preds, ws


# def blur_forest(
#     X,
#     y,
#     eps=None,
#     Chol_t=None,
#     Chol_eps=None,
#     Theta_p=None,
#     Theta_e=None,
#     model=BlurredForest(),
#     rand_type="full",
#     use_expectation=False,
#     est_risk=True,
# ):

#     X, y, model, n, p = _preprocess_X_y_model(X, y, model)

#     full_rand = _get_rand_bool(rand_type)

#     (
#         Chol_t,
#         Sigma_t,
#         Chol_eps,
#         Sigma_eps,
#         Prec_eps,
#         proj_t_eps,
#         Theta_p,
#         Chol_p,
#         Sigma_t_Theta_p,
#         Aperp,
#     ) = _compute_matrices(n, Chol_t, Chol_eps, Theta_p)

#     if Theta_e is not None:
#         Chol_e = np.linalg.cholesky(Theta_e)
#     else:
#         # Theta_e = Prec_eps
#         # Chol_e = np.linalg.cholesky(Theta_e)
#         Chol_e = np.eye(n)
#     X_e = Chol_e.T @ X

#     model.fit(X_e, y, chol_eps=Chol_eps, bootstrap_type="blur")

#     P = model.get_linear_smoother(X)
#     if eps is None:
#         eps = model.eps_
#     else:
#         eps = [eps]

#     n_trees = len(P)

#     tree_ests, centered_preds, ws = get_estimate_terms(
#         y,
#         P,
#         eps,
#         Sigma_t,
#         Theta_p,
#         Chol_p,
#         Sigma_t_Theta_p,
#         proj_t_eps,
#         Aperp,
#         full_rand,
#         use_expectation,
#         est_risk,
#     )
#     return (
#         (
#             tree_ests.sum()
#             # - np.sum(centered_preds**2)) / (n * n_trees), model, ws
#             - np.sum((Chol_e @ centered_preds) ** 2)
#         )
#         / (n * n_trees),
#         model,
#         ws,
#     )


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
