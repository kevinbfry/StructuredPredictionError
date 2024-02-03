from itertools import product
import inspect

import numpy as np

from sklearn.linear_model import LinearRegression, Lasso

from tqdm import tqdm

from spe.relaxed_lasso import RelaxedLasso#, BaggedRelaxedLasso
from spe.cov_estimation import est_Sigma
from spe.tree import Tree
from spe.forest import BlurredForestRegressor
from spe.estimators import (
    kfoldcv,
    kmeanscv,
    by_spatial,
    cp_smoother,
    cp_adaptive_smoother,
    cp_general,
    cp_bagged,
    cp_rf,
    new_y_est,
    bag_kfoldcv,
    bag_kmeanscv,
    simple_train_test_split,
)

from .data_generation import create_clus_split, gen_matern_X, gen_rbf_X ## TODO: gen_rbf_X never used, only imported from here by notebooks. should data_generation not be in package?


class ErrorComparer(object):
    DATA_ARGS = ["X", "y", "y2", "tr_idx", "Chol_t", "Chol_s"]
    BAGCV_METHODS = (bag_kfoldcv, bag_kmeanscv)
    CV_METHODS = (kfoldcv, kmeanscv) + BAGCV_METHODS
    SPCV_METHODS = (bag_kmeanscv, kmeanscv)
    BAGCP_METHODS = (cp_rf, cp_bagged)#, cp_bagged2)
    GENCP_METHODS = (cp_smoother, cp_adaptive_smoother, cp_general) + BAGCP_METHODS
    TESTERR_METHODS = (new_y_est,)

    def gen_X_beta(self, n, p, s, X_kernel=None, c_x=None, c_y=None, ls=None, nu=None):
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

    def gen_mu_sigma(
        self, 
        X, 
        beta, 
        snr, 
        const_mu=False, 
        friedman_mu=False, 
        piecewise_const_mu=False,
        sigma=None
    ):
        if friedman_mu:
            assert(X.shape[1] == 5)
            # (10 sin(πx_1x_2)+ 20(x_3 −0.5)^2 + 10x_4 + 5x_5)/6
            mu = (
                10 * np.sin(X[:,0]*X[:,1]*np.pi) +
                20 * (X[:,2] - 0.5)**2 +
                10 * X[:,3] + 5 * X[:,4]
            ) / 6.
        elif piecewise_const_mu:
            assert(X.shape[1] == 5)
            def get_piecewise_const_mu(X):
                rng = np.random.default_rng(1)
                return (
                    rng.normal() * (X[:,0] > rng.normal())
                    + rng.normal() * (X[:,1] > rng.normal())
                    + rng.normal() * (X[:,2] > rng.normal())
                    + rng.normal() * (X[:,3] > rng.normal())
                    + rng.normal() * (X[:,4] > rng.normal())
                )

            mu = get_piecewise_const_mu(X)
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

        Sigma_t = Chol_t @ Chol_t.T
        Sigma_s = Chol_s @ Chol_s.T
        
        if Cov_st is None:
            eps = Chol_t @ np.random.randn(n)
            eps2 = Chol_s @ np.random.randn(n)
            full_Cov = np.block([
                [Sigma_t, np.zeros_like(Sigma_t)],
                [np.zeros_like(Sigma_t), Sigma_s]
            ])
            self.Chol_f = np.linalg.cholesky(full_Cov)
        else:
            full_Cov = np.block([
                [Sigma_t, Cov_st],
                [Cov_st, Sigma_s]
            ])
            self.Chol_f = Chol_f = np.linalg.cholesky(full_Cov)

            full_eps = Chol_f @ np.random.randn(2*n)
            eps = full_eps[:n]
            eps2 = full_eps[n:]

        y = mu + eps
        y2 = mu + eps2

        return y, y2

    def get_train(self, X, y, coord, tr_idx):
        return X[tr_idx, :], y[tr_idx], coord[tr_idx, :] if len(coord.shape) == 2 else coord[tr_idx]

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
        piecewise_const_mu=False,
        noise_sigma=None,
        risk=False,
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
            mu, sigma = self.gen_mu_sigma(X, beta, snr, const_mu=const_mu, piecewise_const_mu=piecewise_const_mu, friedman_mu=friedman_mu, sigma=noise_sigma)
            Chol_t, Chol_s, Cov_st = self.preprocess_chol(
                self.Chol_y, self.Chol_ystar, sigma, n, Cov_st=self.Cov_y_ystar
            )

        for i in tqdm(range(niter)):
            if gen_beta:
                X, beta = self.gen_X_beta(n, p, s, X_kernel=X_kernel, c_x=coord[:,0], c_y=coord[:,1], ls=X_ls, nu=X_nu)
                mu, sigma = self.gen_mu_sigma(X, beta, snr, const_mu=const_mu, friedman_mu=friedman_mu, piecewise_const_mu=piecewise_const_mu, sigma=noise_sigma)
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
                if self.Chol_ystar is not None:
                    raise ValueError("est_sigma=True not implemented for Chol_ystar != None")

                if est_sigma == 'over':
                    X_over = np.random.randn(n,p)
                    X_est = np.hstack([X, X_over])
                else:
                    X_est = X
                est_covs = est_Sigma(X_est, y, coord, est_sigma, est_sigma_model)
                if est_sigma == 'corr_resp':
                    est_Chol_t = est_covs[0]
                    est_Cov_st = est_covs[1]
                else:
                    est_Chol_t = est_covs
                    est_Cov_st = None
                est_Chol_s = None
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
                elif ests[j]  == by_spatial:
                    est_kwargs[j] = {**est_kwargs[j], **{
                            "X": X, 
                            "Chol_f": self.Chol_f, ## TODO: this is still using truth
                            "y": y, 
                        }
                    }
                elif ests[j]  == simple_train_test_split:
                    est_kwargs[j] = {**est_kwargs[j], **{
                            "X": X, 
                            "y": y, 
                            "tr_idx": tr_idx,
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
                    if not (delta is None): ## TODO: think this can be 'if delta is not None'
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
                if risk:
                    err[i] -= sigma**2

        return errs