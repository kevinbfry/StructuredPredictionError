from itertools import product
import inspect

import numpy as np

# from scipy.linalg import block_diag, sqrtm
# from scipy.spatial.distance import squareform
from scipy.spatial import distance_matrix

from sklearn.linear_model import LinearRegression, Lasso
# from sklearn.model_selection import cross_validate, GroupKFold, KFold
# from sklearn.cluster import KMeans
# from sklearn.base import clone
# from sklearn.ensemble import RandomForestRegressor

import skgstat as skg

from tqdm import tqdm

from spe.relaxed_lasso import RelaxedLasso#, BaggedRelaxedLasso
from spe.estimators import (
    kfoldcv,
    kmeanscv,
    timeseriescv,
    cp_smoother_train_test,
    cp_adaptive_smoother_train_test,
    cp_general_train_test,
    cp_bagged_train_test,
    cp_rf_train_test,
    better_test_est_split,
    ts_test_est_split,
    bag_kfoldcv,
    bag_kmeanscv,
)

from spe.tree import Tree
from spe.forest import BlurredForest
from .data_generation import create_clus_split, gen_matern_X, gen_rbf_X ## TODO: gen_rbf_X never used, only imported from here by notebooks. should data_generation not be in package?


class ErrorComparer(object):
    DATA_ARGS = ["X", "y", "y2", "tr_idx", "Chol_t", "Chol_s"]
    BAGCV_METHODS = (bag_kfoldcv, bag_kmeanscv)
    CV_METHODS = (kfoldcv, kmeanscv, timeseriescv) + BAGCV_METHODS
    SPCV_METHODS = (bag_kmeanscv, kmeanscv)
    BAGCP_METHODS = (cp_rf_train_test, cp_bagged_train_test)#, cp_bagged_train_test2)
    GENCP_METHODS = (cp_smoother_train_test, cp_adaptive_smoother_train_test, cp_general_train_test) + BAGCP_METHODS
    TESTERR_METHODS = (better_test_est_split, ts_test_est_split)

    # BAGCV_METHODS = ["bag_kfoldcv", "bag_kmeanscv"]
    # CV_METHODS = ["kfoldcv", "kmeanscv", 'timeseriescv'] + BAGCV_METHODS
    # SPCV_METHODS = ["bag_kmeanscv", "kmeanscv"]
    # GENCP_METHODS = ['cp_smoother_train_test', 'cp_adaptive_smoother_train_test', 'cp_general_train_test']

    def gen_X_beta(self, n, p, s, X_kernel=None, c_x=None, c_y=None, ls=None, nu=None):
        # X = np.random.randn(n, p)
        if X_kernel == 'rbf':
            X = gen_rbf_X(c_x, c_y, p)
        elif X_kernel == 'matern':
            X = gen_matern_X(c_x, c_y, p, length_scale=ls, nu=nu)
        else:
            X = np.random.randn(n,p)
        beta = np.zeros(p)
        idx = np.random.choice(p, size=s)
        beta[idx] = np.random.uniform(-1, 1, size=s)

        return X, beta

    def gen_mu_sigma(self, X, beta, snr, const_mu=False, friedman_mu=False, sigma=None):
        if friedman_mu:
            assert(X.shape[1] == 5)
            # (10 sin(πx_1x_2)+ 20(x_3 −0.5)2 + 10x_4 + 5x_5)/6
            mu = (
                10 * np.sin(X[:,0]*X[:,1]*np.pi) +
                20 * (X[:,2] - 0.5)**2 +
                10 * X[:,3] + 5 * X[:,4]
            ) / 6.
        elif const_mu:
            mu = np.ones(X.shape[0])*beta.sum()*5
        else:
            mu = X @ beta

        if sigma is None:
            sigma = np.sqrt(np.var(mu) / snr)

        return mu, sigma

    def preprocess_X_beta(self, X, beta, n, p, friedman_mu=False, const_mu=False):
        gen_beta = X is None or (beta is None and (not friedman_mu and not const_mu))
        if X is not None:
            n, p = X.shape
        return gen_beta, n, p

    def preprocess_chol(self, Chol_t, Chol_s, sigma, n, Cov_st=None):
        if Chol_t is None:
            Chol_t = np.eye(n)
        Chol_t *= sigma

        if Chol_s is None:
            Chol_s = Chol_t
        else:
            Chol_s *= sigma

        if Cov_st is not None:
            Cov_st *= sigma**2

        return Chol_t, Chol_s, Cov_st

    def gen_ys(self, mu, Chol_t, Chol_s, sigma=1.0, Cov_st=None, delta=1.): ## TODO: why is delta here?
        n = len(mu)
        if Cov_st is None:
            eps = Chol_t @ np.random.randn(n)
            eps2 = Chol_s @ np.random.randn(n)
        else:
            Sigma_t = Chol_t @ Chol_t.T
            Sigma_s = Chol_s @ Chol_s.T
            full_Cov = np.vstack((
                np.hstack((Sigma_t, Cov_st)),
                np.hstack((Cov_st, Sigma_s))
            ))
            Chol_f = np.linalg.cholesky(full_Cov)

            full_eps = Chol_f @ np.random.randn(2*n)
            eps = full_eps[:n]
            eps2 = full_eps[n:]

        y = mu + eps
        y2 = mu + eps2

        return y, y2

    def get_train(self, X, y, coord, tr_idx):
        return X[tr_idx, :], y[tr_idx], coord[tr_idx, :] if len(coord.shape) == 2 else coord[tr_idx]

    def est_Sigma(
        self,
        # X_tr, 
        # y_tr, 
        # locs_tr,
        X,
        y,
        locs, ## TODO: only needed now to work with how _forest.py eps variable is processed. should fix and then remove this too
        # tr_idx,
        # ts_idx,
        est_sigma_model, 
    ):
        n = locs.shape[0]

        if est_sigma_model is None:
            raise ValueError("Must provide est_simga_model")

        # est_sigma_model.fit(X_tr, y_tr)
        # resids =  y_tr - est_sigma_model.predict(X_tr)
        est_sigma_model.fit(X, y)
        resids =  y - est_sigma_model.predict(X)

        # V = skg.Variogram(locs_tr, resids, model='matern')
        V = skg.Variogram(locs, resids, model='matern')
        
        fitted_vm = V.fitted_model
        full_distance = distance_matrix(locs, locs)
        semivar = fitted_vm(full_distance.flatten()).reshape((n,n))

        K0 = V.parameters[1] ## use sill as estimate of variance
        est_Sigma_full = K0*np.ones_like(semivar) - semivar
        est_Chol_t = np.linalg.cholesky(est_Sigma_full)#[tr_idx,:][:,tr_idx])
        # est_Chol_s = np.linalg.cholesky(est_Sigma_full[ts_idx,:][:,ts_idx])

        return est_Chol_t#, est_Chol_s

    ## TODO: cleanup, compartmentalize
    def compare(
        self,
        models,
        ests,
        est_kwargs,
        niter=100,
        n=200,
        p=30,
        s=5,
        snr=0.4,
        X=None,
        X_kernel=None,
        X_ls=None,
        X_nu=None,
        beta=None,
        coord=None,
        Chol_y=None,
        Chol_ystar=None,
        Cov_y_ystar=None,
        delta=None,
        tr_idx=None,
        fair=False,
        tr_frac=0.6,
        est_sigma=False,
        est_sigma_model=None,
        const_mu=False,
        friedman_mu=False,
        noise_sigma=None,
        **kwargs,
    ):

        self.Chol_y = np.copy(Chol_y) if Chol_y is not None else None
        self.Chol_ystar = np.copy(Chol_ystar) if Chol_ystar is not None else None
        self.Cov_y_ystar = np.copy(Cov_y_ystar) if Cov_y_ystar is not None else None

        if len(ests) != len(est_kwargs):
            raise ValueError("ests must be same length as est_kwargs")

        if not isinstance(models, (list, tuple)):
            models = [models] * len(ests)
        elif len(models) == 1:
            models = models * len(ests)
        elif len(ests) != len(models):
            raise ValueError("ests must be same length as models")

        errs = [np.zeros(niter) for _ in range(len(ests))]
        for j, est in enumerate(ests):
            if est in self.CV_METHODS:
                est_kwargs[j] = {**est_kwargs[j], **kwargs, **{"model": models[j]}}
            else:
                est_kwargs[j]["model"] = models[j]

        gen_beta, n, p = self.preprocess_X_beta(X, beta, n, p, friedman_mu, const_mu)

        if not gen_beta:
            mu, sigma = self.gen_mu_sigma(X, beta, snr, const_mu=const_mu, friedman_mu=friedman_mu, sigma=noise_sigma)
            Chol_t, Chol_s, Cov_st = self.preprocess_chol(
                self.Chol_y, self.Chol_ystar, sigma, n, Cov_st=self.Cov_y_ystar
            )

        for i in tqdm(range(niter)):
            if gen_beta:
                X, beta = self.gen_X_beta(n, p, s, X_kernel=X_kernel, c_x=coord[:,0], c_y=coord[:,1], ls=X_ls, nu=X_nu)
                mu, sigma = self.gen_mu_sigma(X, beta, snr, const_mu=const_mu, friedman_mu=friedman_mu, sigma=noise_sigma)
                Chol_t, Chol_s, Cov_st = self.preprocess_chol(
                    self.Chol_y, self.Chol_ystar, sigma, n, Cov_st=self.Cov_y_ystar
                )

            if tr_idx is None:
                if fair:
                    tr_samples = np.random.choice(
                        n, size=int(tr_frac * n), replace=False
                    )
                    tr_idx = np.zeros(n).astype(bool)
                    tr_idx[tr_samples] = True
                else:
                    tr_idx = create_clus_split(
                        int(np.sqrt(n)), int(np.sqrt(n)), tr_frac
                    )

            y, y2 = self.gen_ys(
                mu, Chol_t, Chol_s, sigma=sigma, Cov_st=Cov_st, delta=delta
            )

            if not fair:
                X_tr, y_tr, coord_tr = self.get_train(X, y, coord, tr_idx)
                cvChol_t = Chol_t[tr_idx, :][:, tr_idx]

            if est_sigma:
                est_Chol_t = self.est_Sigma(X, y, coord, est_sigma_model)
                if self.Chol_ystar is not None:
                    raise ValueError("est_sigma=True not implemented for Chol_s != None")
                if self.Cov_y_ystar is not None:
                    raise ValueError("est_sigma=True not implemented for Cov_st != None")
                est_Chol_s = est_Chol_t
                est_Cov_st = None
            else:
                est_Chol_t = np.copy(Chol_t)
                est_Chol_s = np.copy(Chol_s) if Chol_s is not None else None
                est_Cov_st = np.copy(Cov_st) if Cov_st is not None else None
            
            for j in range(len(est_kwargs)):
                if ests[j] in self.TESTERR_METHODS:
                    est_kwargs[j] = {**est_kwargs[j], **{
                            "X": X, 
                            "Chol_t": est_Chol_t, 
                            "tr_idx": tr_idx, 
                            "y": y, 
                            "y2": y2
                        }
                    }
                else:
                    est_kwargs[j] = {
                        **est_kwargs[j],
                        **{"X": X, 
                            "Chol_t": est_Chol_t, 
                            "Chol_s": est_Chol_s, 
                            "tr_idx": tr_idx, 
                            "y": y
                        },
                    }
                    if not (delta is None):
                        if ests[j] in self.CV_METHODS:
                            est_kwargs[j] = {**est_kwargs[j], **{"Cov_st": est_Cov_st}}

                if ests[j] in self.GENCP_METHODS:
                    est_kwargs[j] = {
                        **est_kwargs[j],
                        **{"Cov_st": est_Cov_st}
                    }

            for j, est in enumerate(ests):
                if est in self.CV_METHODS:
                    if fair:
                        if est in self.SPCV_METHODS:
                            est_kwargs[j]["coord"] = coord
                    else:
                        est_kwargs[j]["X"] = X_tr
                        est_kwargs[j]["y"] = y_tr
                        if est in self.BAGCV_METHODS:
                            est_kwargs[j]["Chol_t"] = cvChol_t
                        if est in self.SPCV_METHODS:
                            est_kwargs[j]["coord"] = coord_tr
                    if est not in self.BAGCV_METHODS:
                        est_kwargs[j].pop("Chol_t", None)
                    est_kwargs[j].pop("Chol_s", None)
                    est_kwargs[j].pop("tr_idx", None)

            for err, est, est_kwarg in zip(errs, ests, est_kwargs):
                err[i] = est(**est_kwarg)

        return errs