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

from spe.relaxed_lasso import RelaxedLasso, BaggedRelaxedLasso
from spe.estimators import (
    kfoldcv,
    kmeanscv,
    test_set_estimator,
    cb,
    cb_isotropic,
    blur_linear,
    blur_linear_selector,
    blur_forest,
    cp_linear_train_test,
    test_est_split,
    cp_relaxed_lasso_train_test,
    cp_bagged_train_test,
    cp_rf_train_test,
    better_test_est_split,
    bag_kfoldcv,
    bag_kmeanscv,
)
from spe.tree import Tree
from spe.forest import BlurredForest
from data_generation import create_clus_split ## TODO: gen_rbf_X never used, only imported from here by notebooks. should data_generation not be in package?


class ErrorComparer(object):
    DATA_ARGS = ["X", "y", "y2", "tr_idx", "Chol_t", "Chol_s"]
    BAGCV_METHODS = ["bag_kfoldcv", "bag_kmeanscv"]
    CV_METHODS = ["kfoldcv", "kmeanscv", 'timeseriescv'] + BAGCV_METHODS
    SPCV_METHODS = ["bag_kmeanscv", "kmeanscv"]

    def gen_X_beta(self, n, p, s):
        X = np.random.randn(n, p)
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
        if not X is None:
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
        X_tr, 
        y_tr, 
        locs_tr,
        locs, ## TODO: only needed now to work with how _forest.py eps variable is processed. should fix and then remove this too
        est_sigma_model, 
    ):
        n = locs.shape[0]

        if est_sigma_model is None:
            raise ValueError("Must provide est_simga_model")

        est_sigma_model.fit(X_tr, y_tr)
        resids =  y_tr - est_sigma_model.predict(X_tr)

        V = skg.Variogram(locs_tr, resids, model='matern')
        
        fitted_vm = V.fitted_model
        full_distance = distance_matrix(locs, locs)
        semivar = fitted_vm(full_distance.flatten()).reshape((n,n))

        K0 = V.parameters[1] ## use sill as estimate of variance
        est_Sigma_t = K0*np.ones_like(semivar) - semivar
        est_Chol_t = np.linalg.cholesky(est_Sigma_t)

        return est_Chol_t

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
        beta=None,
        coord=None,
        Chol_t=None,
        Chol_s=None,
        Cov_st=None,
        delta=None,
        tr_idx=None,
        fair=False,
        tr_frac=0.6,
        est_sigma=False,
        est_sigma_model=None,
        const_mu=False,
        friedman_mu=False,
        sigma=None,
        # test_kwargs={},
        **kwargs,
    ):
        # print("NEW NEW")

        if len(ests) != len(est_kwargs):
            raise ValueError("ests must be same length as est_kwargs")

        if not isinstance(models, (list, tuple)):
            models = [models] * len(ests)
        elif len(models) == 1:
            models = models * len(ests)
        elif len(ests) != len(models):
            raise ValueError("ests must be same length as models")

        errs = [np.zeros(niter) for _ in range(len(ests))]
        # errs = [np.zeros(niter) for _ in range(len(ests)+1)]
        # ests.insert(0, better_test_est_split)
        # est_kwargs.insert(0, test_kwargs)
        # print(models)
        for j, est in enumerate(ests):
            if est.__name__ not in self.CV_METHODS:
                est_kwargs[j] = {**est_kwargs[j], **kwargs, **{"model": models[j]}}
            else:
                est_kwargs[j]["model"] = models[j]
            # if j == 0:
            # 	est_kwargs[j]['model'] = model

        gen_beta, n, p = self.preprocess_X_beta(X, beta, n, p, friedman_mu, const_mu)

        Chol_t_orig = Chol_t
        Chol_s_orig = Chol_s
        Cov_st_orig = Cov_st

        if not gen_beta:
            mu, sigma = self.gen_mu_sigma(X, beta, snr, const_mu=const_mu, friedman_mu=friedman_mu, sigma=sigma)
            Chol_t, Chol_s, Cov_st = self.preprocess_chol(
                Chol_t_orig, Chol_s_orig, sigma, n, Cov_st=Cov_st_orig
            )

        for i in np.arange(niter):
            if i % 10 == 0:
                print(i)

            if gen_beta:
                X, beta = self.gen_X_beta(n, p, s)
                mu, sigma = self.gen_mu_sigma(X, beta, snr, const_mu=const_mu, friedman_mu=friedman_mu, sigma=sigma)
                Chol_t, Chol_s, Cov_st = self.preprocess_chol(
                    Chol_t_orig, Chol_s_orig, sigma, n, Cov_st=Cov_st_orig
                )

            if tr_idx is None:
                if fair:
                    # tr_samples = np.random.choice(n, size=int(.8*n), replace=False)
                    tr_samples = np.random.choice(
                        n, size=int(tr_frac * n), replace=False
                    )
                    tr_idx = np.zeros(n).astype(bool)
                    tr_idx[tr_samples] = True
                else:
                    tr_idx = create_clus_split(
                        int(np.sqrt(n)), int(np.sqrt(n)), tr_frac
                    )
            if i == 0:
                print(tr_idx.mean())

            y, y2 = self.gen_ys(
                mu, Chol_t, Chol_s, sigma=sigma, Cov_st=Cov_st, delta=delta
            )

            if not fair:
                X_tr, y_tr, coord_tr = self.get_train(X, y, coord, tr_idx)
                cvChol_t = Chol_t[tr_idx, :][:, tr_idx]

            if est_sigma:
                est_Chol_t = self.est_Sigma(X_tr, y_tr, coord_tr, coord, est_sigma_model)
                if Chol_s_orig is not None:
                    raise ValueError("est_sigma=True not implemented for Chol_s != None")
                if Cov_st_orig is not None:
                    raise ValueError("est_sigma=True not implemented for Cov_st != None")
                est_Chol_s = est_Chol_t
            else:
                est_Chol_t = Chol_t
                est_Chol_s = Chol_s
                est_Cov_st = Cov_st
            
            for j in range(len(est_kwargs)):
                if ests[j].__name__ in ["better_test_est_split", "ts_test_est_split"]:
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
                        if ests[j].__name__ not in self.CV_METHODS:
                            est_kwargs[j] = {**est_kwargs[j], **{"Cov_st": est_Cov_st}}

                if est_sigma and ests[j].__name__ == 'cp_rf_train_test':
                    est_kwargs[j]['chol_eps'] = est_Chol_t

            for j, est in enumerate(ests):
                if est.__name__ in self.CV_METHODS:
                    if fair:
                        if est.__name__ in self.SPCV_METHODS:
                            est_kwargs[j]["coord"] = coord
                    else:
                        est_kwargs[j]["X"] = X_tr
                        est_kwargs[j]["y"] = y_tr
                        if est.__name__ in self.BAGCV_METHODS:
                            est_kwargs[j]["Chol_t"] = cvChol_t
                        if est.__name__ in self.SPCV_METHODS:
                            est_kwargs[j]["coord"] = coord_tr
                    if est.__name__ not in self.BAGCV_METHODS:
                        est_kwargs[j].pop("Chol_t", None)
                    est_kwargs[j].pop("Chol_s", None)
                    est_kwargs[j].pop("tr_idx", None)

            for err, est, est_kwarg in zip(errs, ests, est_kwargs):
                err[i] = est(**est_kwarg)

        return errs

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
        Chol_s=None,
        tr_idx=None,
        tr_frac=.6,
        const_mu=False,
        friedman_mu=False,
        k=10,
    ):
        return self.compare(
            LinearRegression(fit_intercept=False),
            [better_test_est_split, kfoldcv, kmeanscv, cp_linear_train_test],
            [{}, {"k": k}, {"k": k}, {}],
            niter=niter,
            n=n,
            p=p,
            s=s,
            snr=snr,
            X=X,
            beta=beta,
            coord=coord,
            Chol_t=Chol_t,
            Chol_s=Chol_s,
            tr_idx=tr_idx,
            tr_frac=tr_frac,
            fair=False,
            const_mu=const_mu,
            friedman_mu=friedman_mu,
            **{},
        )

    def compareLinearTrTsFair(
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
        Chol_s=None,
        tr_idx=None,
        k=10,
    ):
        return self.compare(
            LinearRegression(fit_intercept=False),
            [better_test_est_split, kfoldcv, kmeanscv, cp_linear_train_test],
            [{}, {"k": k}, {"k": k}, {}],
            niter=niter,
            n=n,
            p=p,
            s=s,
            snr=snr,
            X=X,
            beta=beta,
            coord=coord,
            Chol_t=Chol_t,
            Chol_s=Chol_s,
            tr_idx=tr_idx,
            fair=True,
            **{},
        )

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
        Chol_s=None,
        alpha=1.0,
        lambd=0.31,
        tr_idx=None,
        tr_frac=.6,
        k=10,
    ):
        return self.compare(
            RelaxedLasso(lambd=lambd),
            [better_test_est_split, kfoldcv, kmeanscv, cp_relaxed_lasso_train_test],
            [{}, {"k": k}, {"k": k}, {"alpha": alpha, "use_trace_corr": True}],
            niter=niter,
            n=n,
            p=p,
            s=s,
            snr=snr,
            X=X,
            beta=beta,
            coord=coord,
            Chol_t=Chol_t,
            Chol_s=Chol_s,
            tr_idx=tr_idx,
            tr_frac=tr_frac,
            fair=False,
        )

    def compareRelaxedLassoTrTsFair(
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
        Chol_s=None,
        alpha=1.0,
        lambd=0.31,
        tr_idx=None,
        k=10,
    ):
        return self.compare(
            RelaxedLasso(lambd=lambd),
            [better_test_est_split, kfoldcv, kmeanscv, cp_relaxed_lasso_train_test],
            [{}, {"k": k}, {"k": k}, {"alpha": alpha, "use_trace_corr": True}],
            niter=niter,
            n=n,
            p=p,
            s=s,
            snr=snr,
            X=X,
            beta=beta,
            coord=coord,
            Chol_t=Chol_t,
            Chol_s=Chol_s,
            tr_idx=tr_idx,
            fair=True,
        )

    def compareBaggedTrTs(
        self,
        base_estimator=RelaxedLasso(lambd=0.1, fit_intercept=False),
        niter=100,
        n=200,
        p=30,
        s=5,
        snr=0.4,
        X=None,
        beta=None,
        coord=None,
        Chol_t=None,
        Chol_s=None,
        n_estimators=10,
        # lambd=0.31,
        tr_idx=None,
        tr_frac=.6,
        k=10,
        **kwargs,
    ):
        return self.compare(
            BaggedRelaxedLasso(
                base_estimator=base_estimator, n_estimators=n_estimators
            ),
            [better_test_est_split, bag_kfoldcv, bag_kmeanscv, cp_bagged_train_test],
            [{}, {"k": k}, {"k": k}, {"use_trace_corr": True}],
            niter=niter,
            n=n,
            p=p,
            s=s,
            snr=snr,
            X=X,
            beta=beta,
            coord=coord,
            Chol_t=Chol_t,
            Chol_s=Chol_s,
            tr_idx=tr_idx,
            tr_frac=tr_frac,
            fair=False,
        )

    def compareBaggedTrTsFair(
        self,
        base_estimator=RelaxedLasso(lambd=0.1, fit_intercept=False),
        niter=100,
        n=200,
        p=30,
        s=5,
        snr=0.4,
        X=None,
        beta=None,
        coord=None,
        Chol_t=None,
        Chol_s=None,
        n_estimators=10,
        # lambd=0.31,
        tr_idx=None,
        k=10,
        **kwargs,
    ):
        return self.compare(
            BaggedRelaxedLasso(
                base_estimator=base_estimator, n_estimators=n_estimators
            ),
            [better_test_est_split, bag_kfoldcv, bag_kmeanscv, cp_bagged_train_test],
            [{}, {"k": k}, {"k": k}, {"use_trace_corr": True}],
            niter=niter,
            n=n,
            p=p,
            s=s,
            snr=snr,
            X=X,
            beta=beta,
            coord=coord,
            Chol_t=Chol_t,
            Chol_s=Chol_s,
            tr_idx=tr_idx,
            fair=True,
        )

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
        Chol_s=None,
        max_depth=4,
        n_estimators=5,
        tr_idx=None,
        tr_frac=.6,
        k=10,
        **kwargs,
    ):
        return self.compare(
            BlurredForest(n_estimators=n_estimators),
            [better_test_est_split, bag_kfoldcv, bag_kmeanscv, cp_rf_train_test],
            [{}, {"k": k}, {"k": k}, {"use_trace_corr": True}],
            niter=niter,
            n=n,
            p=p,
            s=s,
            snr=snr,
            X=X,
            beta=beta,
            coord=coord,
            Chol_t=Chol_t,
            Chol_s=Chol_s,
            tr_idx=tr_idx,
            tr_frac=tr_frac,
            fair=False,
            **kwargs,
        )

    def compareForestTrTsFair(
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
        Chol_s=None,
        max_depth=4,
        n_estimators=5,
        tr_idx=None,
        k=10,
        **kwargs,
    ):
        return self.compare(
            BlurredForest(n_estimators=n_estimators),
            [better_test_est_split, bag_kfoldcv, bag_kmeanscv, cp_rf_train_test],
            [{}, {"k": k}, {"k": k}, {"use_trace_corr": True}],
            niter=niter,
            n=n,
            p=p,
            s=s,
            snr=snr,
            X=X,
            beta=beta,
            coord=coord,
            Chol_t=Chol_t,
            Chol_s=Chol_s,
            tr_idx=tr_idx,
            fair=True,
            **kwargs,
        )
    

    




    