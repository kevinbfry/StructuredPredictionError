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
from spe.data_generation import gen_rbf_X, create_clus_split


class ErrorComparer(object):
    DATA_ARGS = ["X", "y", "y2", "tr_idx", "Chol_t", "Chol_s"]
    BAGCV_METHODS = ["bag_kfoldcv", "bag_kmeanscv"]
    CV_METHODS = ["kfoldcv", "kmeanscv", 'timeseriescv'] + BAGCV_METHODS
    SPCV_METHODS = ["bag_kmeanscv", "kmeanscv"]

    def _gen_X_beta(self, n, p, s):
        X = np.random.randn(n, p)
        beta = np.zeros(p)
        idx = np.random.choice(p, size=s)
        beta[idx] = np.random.uniform(-1, 1, size=s)

        return X, beta

    def _gen_mu_sigma(self, X, beta, snr, const_mu=False, friedman_mu=False, sigma=None):
        if friedman_mu:
            assert(X.shape[1] == 5)
            # (10 sin(πx1x2)+ 20(x3 −0.5)2 + 10x4 + 5x5)/6
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

    def _preprocess_X_beta(self, X, beta, n, p):
        gen_beta = X is None or beta is None
        if not X is None:
            n, p = X.shape
        return gen_beta, n, p

    def _preprocess_chol(self, Chol_t, Chol_s, sigma, n, Cov_st=None):
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

    def _gen_ys(self, mu, Chol_t, Chol_s, sigma=1.0, Cov_st=None, delta=1.):
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

            # shared_eps = np.linalg.cholesky(Cov_st) @ np.random.randn(n)
            # eps = shared_eps + np.sqrt(1 - delta) * sigma * np.random.randn(n)
            # eps2 = shared_eps + np.sqrt(1 - delta) * sigma * np.random.randn(n)
        
        y = mu + eps
        y2 = mu + eps2

        return y, y2

    def _get_train(self, X, y, coord, tr_idx):
        return X[tr_idx, :], y[tr_idx], coord[tr_idx, :] if len(coord.shape) == 2 else coord[tr_idx]

    def _est_Sigma(
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

        gen_beta, n, p = self._preprocess_X_beta(X, beta, n, p)

        Chol_t_orig = Chol_t
        Chol_s_orig = Chol_s
        Cov_st_orig = Cov_st

        if not gen_beta:
            mu, sigma = self._gen_mu_sigma(X, beta, snr, const_mu=const_mu, friedman_mu=friedman_mu, sigma=sigma)
            Chol_t, Chol_s, Cov_st = self._preprocess_chol(
                Chol_t_orig, Chol_s_orig, sigma, n, Cov_st=Cov_st_orig
            )

            # if est_sigma:
            #     est_Chol_t = self._est_Sigma(X, y, coord, est_sigma_model)
            #     if Chol_s is not None:
            #         raise ValueError("est_sigma not implemented for Chol_s != None")
            #     if Cov_st is not None:
            #         raise ValueError("est_sigma not implemented for Cov_st != None")
            #     est_Chol_s = est_Chol_t
            # else:
            #     est_Chol_t = Chol_t
            #     est_Chol_s = Chol_s
            #     est_Cov_st = Cov_st

            # for j in range(len(est_kwargs)):
            #     # if j == 0:
            #     if ests[j].__name__ == "better_test_est_split":
            #         est_kwargs[j] = {**est_kwargs[j], **{"X": X, "Chol_t": est_Chol_t}}
            #     else:
            #         est_kwargs[j] = {
            #             **est_kwargs[j],
            #             **{"X": X, "Chol_t": est_Chol_t, "Chol_s": est_Chol_s},
            #         }
            #         if delta is not None:
            #             if ests[j].__name__ not in self.CV_METHODS:
            #                 est_kwargs[j] = {**est_kwargs[j], **{"Cov_st": est_Cov_st}}
                # print(est_kwargs[j].keys())

        for i in np.arange(niter):
            if i % 10 == 0:
                print(i)
            # print(i)

            if gen_beta:
                X, beta = self._gen_X_beta(n, p, s)
                mu, sigma = self._gen_mu_sigma(X, beta, snr, const_mu=const_mu, friedman_mu=friedman_mu, sigma=sigma)
                Chol_t, Chol_s, Cov_st = self._preprocess_chol(
                    Chol_t_orig, Chol_s_orig, sigma, n, Cov_st=Cov_st_orig
                )

                # for j in range(len(est_kwargs)):
                #     # if j == 0:
                #     if ests[j].__name__ == "better_test_est_split":
                #         est_kwargs[j] = {**est_kwargs[j], **{"X": X, "Chol_t": est_Chol_t}}
                #     else:
                #         est_kwargs[j] = {
                #             **est_kwargs[j],
                #             **{"X": X, "Chol_t": est_Chol_t, "Chol_s": est_Chol_s},
                #         }
                #         if not (delta is None):
                #             if ests[j].__name__ not in self.CV_METHODS:
                #                 est_kwargs[j] = {**est_kwargs[j], **{"Cov_st": est_Cov_st}}

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

            y, y2 = self._gen_ys(
                mu, Chol_t, Chol_s, sigma=sigma, Cov_st=Cov_st, delta=delta
            )

            # # print("after y2")
            # for j in range(len(est_kwargs)):
            #     # if j == 0:
            #     if ests[j].__name__ == "better_test_est_split":
            #         est_kwargs[j] = {
            #             **est_kwargs[j],
            #             **{"tr_idx": tr_idx, "y": y, "y2": y2},
            #         }
            #     else:
            #         est_kwargs[j] = {**est_kwargs[j], **{"tr_idx": tr_idx, "y": y}}

            #     # print(est_kwargs[j].keys())

            if not fair:
                X_tr, y_tr, coord_tr = self._get_train(X, y, coord, tr_idx)
                cvChol_t = Chol_t[tr_idx, :][:, tr_idx]

            if est_sigma:
                est_Chol_t = self._est_Sigma(X_tr, y_tr, coord_tr, coord, est_sigma_model)
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
                # if j == 0:
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

            # print(ests)
            # print([e.keys() for e in est_kwargs])
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

    ## TODO: write new version of this that calls better_test_est_split
    ##       and calls compare(). 
    def compareGLSFTrTs(
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
        const_mu=False,
        **kwargs,
    ):

        # model = RelaxedLasso(lambd=lambd, fit_intercept=False)
        model = BlurredForest(
            max_depth=max_depth,
            n_estimators=n_estimators,
        )

        self.test_err = np.zeros(niter)
        # self.kfcv_err = np.zeros(niter)
        # self.spcv_err = np.zeros(niter)
        self.bagg_err = np.zeros(niter)
        self.glsf_err = np.zeros(niter)

        test_est = better_test_est_split
        # kfcv_est = bag_kfoldcv
        # spcv_est = bag_kmeanscv
        bagg_est = cp_rf_train_test
        glsf_est = cp_rf_train_test

        gen_beta, n, p = self._preprocess_X_beta(X, beta, n, p)

        Chol_t_orig = Chol_t
        Chol_s_orig = Chol_s

        if not gen_beta:
            mu, sigma = self._gen_mu_sigma(X, beta, snr, const_mu=const_mu)

            Chol_t, Chol_s, _ = self._preprocess_chol(
                Chol_t_orig, Chol_s_orig, sigma, n
            )

            kwargs["chol_eps"] = Chol_t

        for i in np.arange(niter):
            if i % 10 == 0:
                print(i)

            if gen_beta:
                X, beta = self._gen_X_beta(n, p, s)
                mu, sigma = self._gen_mu_sigma(X, beta, snr,const_mu=const_mu)
                Chol_t, Chol_s = self._preprocess_chol(
                    Chol_t_orig, Chol_s_orig, sigma, n
                )

                kwargs["chol_eps"] = Chol_t

            tr_idx = create_clus_split(int(np.sqrt(n)), int(np.sqrt(n)), tr_frac=tr_frac)
            # tr_samples = np.random.choice(n, size=int(.8*n), replace=False)
            # tr_idx = np.zeros(n).astype(bool)
            # tr_idx[tr_samples] = True
            # ts_idx = (1 - tr_idx).astype(bool)
            # kwargs["idx_tr"] = tr_idx
            if i == 0:
                print(tr_idx.mean())

            # if tr_idx is None:
            # tr_samples = np.random.choice(n, size=int(.8*n), replace=False)
            # tr_idx = np.zeros(n)
            # tr_idx[tr_samples] = 1
            # ts_idx = (1 - tr_idx).astype(bool)

            y, y2 = self._gen_ys(mu, Chol_t, Chol_s)

            X_tr, y_tr, coord_tr = self._get_train(X, y, coord, tr_idx)

            self.test_err[i] = test_est(
                model=model, X=X, y=y, y2=y2, tr_idx=tr_idx, **kwargs
                # model=model, X=X, y=y, y2=y2, **kwargs
            )

            self.bagg_err[i], self.glsf_err[i] = bagg_est(
                model=model,
                X=X,
                y=y,
                tr_idx=tr_idx,
                Chol_t=Chol_t,
                Chol_s=Chol_s,
                n_estimators=n_estimators,
                ret_gls=True,
                **kwargs,
            )

            # cvChol_t = Chol_t[tr_idx,:][:,tr_idx]

            # self.kfcv_err[i] = kfcv_est(model=model,
            # 							X=X_tr,
            # 							y=y_tr,
            # 							k=k,
            # 							Chol_t=cvChol_t)

            # self.spcv_err[i] = spcv_est(model=model,
            # 							X=X_tr,
            # 							y=y_tr,
            # 							coord=coord_tr,
            # 							k=k,
            # 							Chol_t=cvChol_t)

        return self.test_err, self.bagg_err, self.glsf_err

    def compareGLSFTrTsFair(
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
        const_mu=False,
        **kwargs,
    ):

        # model = RelaxedLasso(lambd=lambd, fit_intercept=False)
        model = BlurredForest(
            max_depth=max_depth,
            n_estimators=n_estimators,
        )

        self.test_err = np.zeros(niter)
        # self.kfcv_err = np.zeros(niter)
        # self.spcv_err = np.zeros(niter)
        self.bagg_err = np.zeros(niter)
        self.glsf_err = np.zeros(niter)

        test_est = better_test_est_split
        # kfcv_est = bag_kfoldcv
        # spcv_est = bag_kmeanscv
        bagg_est = cp_rf_train_test
        glsf_est = cp_rf_train_test

        gen_beta, n, p = self._preprocess_X_beta(X, beta, n, p)

        Chol_t_orig = Chol_t
        Chol_s_orig = Chol_s

        if not gen_beta:
            mu, sigma = self._gen_mu_sigma(X, beta, snr, const_mu=const_mu)

            Chol_t, Chol_s = self._preprocess_chol(Chol_t_orig, Chol_s_orig, sigma, n)

            kwargs["chol_eps"] = Chol_t

        for i in np.arange(niter):
            if i % 10 == 0:
                print(i)

            if gen_beta:
                X, beta = self._gen_X_beta(n, p, s)
                mu, sigma = self._gen_mu_sigma(X, beta, snr, const_mu=const_mu)
                Chol_t, Chol_s = self._preprocess_chol(
                    Chol_t_orig, Chol_s_orig, sigma, n
                )

                kwargs["chol_eps"] = Chol_t

            # tr_idx = create_clus_split(int(np.sqrt(n)), int(np.sqrt(n)))
            tr_samples = np.random.choice(n, size=int(0.8 * n), replace=False)
            tr_idx = np.zeros(n).astype(bool)
            tr_idx[tr_samples] = True
            ts_idx = (1 - tr_idx).astype(bool)
            kwargs["idx_tr"] = tr_idx
            if i == 0:
                print(tr_idx.mean())

            # if tr_idx is None:
            # tr_samples = np.random.choice(n, size=int(.8*n), replace=False)
            # tr_idx = np.zeros(n)
            # tr_idx[tr_samples] = 1
            # ts_idx = (1 - tr_idx).astype(bool)

            y, y2 = self._gen_ys(mu, Chol_t, Chol_s)

            X_tr, y_tr, coord_tr = self._get_train(X, y, coord, tr_idx)

            self.test_err[i] = test_est(
                model=model, X=X, y=y, y2=y2, tr_idx=tr_idx, **kwargs
            )

            self.bagg_err[i], self.glsf_err[i] = bagg_est(
                model=model,
                X=X,
                y=y,
                tr_idx=tr_idx,
                Chol_t=Chol_t,
                Chol_s=Chol_s,
                n_estimators=n_estimators,
                ret_gls=True,
                **kwargs,
            )

            # self.kfcv_err[i] = kfcv_est(model=model,
            # 							X=X,
            # 							y=y,
            # 							k=k,
            # 							Chol_t=Chol_t)

            # self.spcv_err[i] = spcv_est(model=model,
            # 							X=X,
            # 							y=y,
            # 							coord=coord,
            # 							k=k,
            # 							Chol_t=Chol_t)

        return self.test_err, self.bagg_err, self.glsf_err

    def compareIID(
        self,
        niter=100,
        n=100,
        p=200,
        s=5,
        snr=0.4,
        X=None,
        beta=None,
        model=Lasso(),
        alpha=0.05,
        est_risk=True,
    ):

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
            sigma = np.sqrt(np.var(mu) / snr)
            n, p = X.shape

        for i in np.arange(niter):

            if gen_beta:
                X = np.random.randn(n, p)
                beta = np.zeros(p)
                idx = np.random.choice(p, size=s)
                beta[idx] = np.random.uniform(-1, 1, size=s)

                mu = X @ beta
                sigma = np.sqrt(np.var(mu) / snr)

            y = mu + sigma * np.random.randn(n)
            y_test = mu + sigma * np.random.randn(n)
            y_alpha = mu + sigma * np.sqrt(1 + alpha) * np.random.randn(n)
            y_test_alpha = mu + sigma * np.sqrt(1 + alpha) * np.random.randn(n)

            self.test_err[i] = test_est(
                model=model,
                X=X,
                y=y,
                y_test=y_test,
                Chol_t=np.eye(n) * sigma,
                est_risk=est_risk,
            )[0]
            self.test_err_alpha[i] = test_est(
                model=model,
                X=X,
                y=y_alpha,
                y_test=y_test_alpha,
                Chol_t=np.eye(n) * np.sqrt(1 + alpha) * sigma,
                est_risk=est_risk,
            )[0]
            self.cb_err[i] = cb_est(
                X=X,
                y=y,
                Chol_t=np.eye(n) * sigma,
                Chol_eps=np.eye(n) * np.sqrt(alpha) * sigma,
                model=model,
                est_risk=est_risk,
            )[0]
            self.cbiso_err[i] = cbiso_est(
                X, y, sigma=sigma, alpha=alpha, model=model, est_risk=est_risk
            )[0]

        return (self.test_err, self.test_err_alpha, self.cb_err, self.cbiso_err)

    def compareLinearSelectorIID(
        self,
        niter=100,
        n=100,
        p=200,
        s=5,
        snr=0.4,
        X=None,
        beta=None,
        model=Tree(),
        rand_type="full",
        use_expectation=False,
        alpha=0.05,
        est_sigma=False,
        est_risk=True,
    ):

        self.test_err = np.zeros(niter)
        self.blur_err = np.zeros(niter)

        test_est = test_set_estimator
        blur_est = blur_linear_selector

        gen_beta = X is None or beta is None

        if not gen_beta:
            mu = X @ beta
            sigma_true = np.sqrt(np.var(mu) / snr)
            n, p = X.shape

        for i in np.arange(niter):

            if gen_beta:
                X = np.random.randn(n, p)
                beta = np.zeros(p)
                idx = np.random.choice(p, size=s)
                beta[idx] = np.random.uniform(-1, 1, size=s)

                mu = X @ beta
                sigma_true = np.sqrt(np.var(mu) / snr)

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

                model.fit(X, y)
                P = model.get_linear_smoother(X)
                df = np.diag(P).sum()
                yhat = P @ y
                sigma = np.sum((y - yhat) ** 2) / (n - df)

            else:
                sigma = sigma_true

            (self.blur_err[i], fitted_model, w) = blur_est(
                X,
                y,
                Chol_t=np.eye(n) * sigma,
                Chol_eps=np.eye(n) * np.sqrt(alpha) * sigma,
                model=model,
                rand_type=rand_type,
                use_expectation=use_expectation,
                est_risk=est_risk,
            )

            G = fitted_model.get_group_X(X)
            y_fit = y if rand_type == "full" else w
            self.test_err[i] = test_est(
                model=LinearRegression(),
                X=G,
                y=y_fit,
                y_test=y_test,
                Chol_t=np.eye(n) * sigma,
                est_risk=est_risk,
            )[0]

        return (self.test_err, self.blur_err)

    def compareLinearSelector(
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
        rand_type="full",
        use_expectation=False,
        alpha=0.05,
        est_sigma=False,
        est_risk=True,
    ):

        self.test_err = np.zeros(niter)
        self.blur_err = np.zeros(niter)

        test_est = test_set_estimator
        blur_est = blur_linear_selector

        gen_beta = X is None or beta is None

        if not gen_beta:
            mu = X @ beta
            sigma_true = np.sqrt(np.var(mu) / snr)
            n, p = X.shape

        if Chol_t is None:
            Chol_t = np.eye(n)
        if Theta_p is not None:
            Theta_p = Theta_p / (sigma_true**2)

        for i in np.arange(niter):

            if gen_beta:
                X = np.random.randn(n, p)
                beta = np.zeros(p)
                idx = np.random.choice(p, size=s)
                beta[idx] = np.random.uniform(-1, 1, size=s)

                mu = X @ beta
                sigma_true = np.sqrt(np.var(mu) / snr)

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

                model.fit(X, y)
                P = model.get_linear_smoother(X)
                df = np.diag(P).sum()
                yhat = P @ y
                sigma = np.sum((y - yhat) ** 2) / (n - df)

            else:
                sigma = sigma_true

            (self.blur_err[i], fitted_model, w) = blur_est(
                X,
                y,
                Chol_t=Chol_t * sigma,
                Chol_eps=Chol_t * np.sqrt(alpha) * sigma,
                Theta_p=Theta_p,
                model=model,
                rand_type=rand_type,
                use_expectation=use_expectation,
                est_risk=est_risk,
            )

            G = fitted_model.get_group_X(X)
            y_fit = y if rand_type == "full" else w
            self.test_err[i] = test_est(
                model=LinearRegression(),
                X=G,
                y=y_fit,
                y_test=y_test,
                Chol_t=Chol_t * sigma,
                # Chol_eps=Chol_t*np.sqrt(alpha)*sigma,
                Theta_p=Theta_p,
                est_risk=est_risk,
            )[0]

        return (self.test_err, self.blur_err)

    def compareForestIID(
        self,
        niter=100,
        n=100,
        p=200,
        s=5,
        snr=0.4,
        X=None,
        beta=None,
        model=Tree(),
        rand_type="full",
        use_expectation=False,
        alpha=0.05,
        est_sigma=False,
        est_risk=True,
    ):

        self.test_err = np.zeros(niter)
        self.blur_err = np.zeros(niter)
        # self.tree_err = np.zeros(niter)

        test_est = test_set_estimator
        blur_est = blur_forest
        # tree_est = blur_linear_selector

        gen_beta = X is None or beta is None

        if not gen_beta:
            mu = X @ beta
            sigma_true = np.sqrt(np.var(mu) / snr)
            n, p = X.shape

        for i in np.arange(niter):

            if gen_beta:
                X = np.random.randn(n, p)
                beta = np.zeros(p)
                idx = np.random.choice(p, size=s)
                beta[idx] = np.random.uniform(-1, 1, size=s)

                mu = X @ beta
                sigma_true = np.sqrt(np.var(mu) / snr)

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

                model.fit(X, y)
                P = model.get_linear_smoother(X)
                df = np.diag(P).sum()
                yhat = P @ y
                sigma = np.sum((y - yhat) ** 2) / (n - df)

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

            (self.blur_err[i], fitted_model, w) = blur_est(
                X,
                y,
                # eps=w - y,
                Chol_t=np.eye(n) * sigma,
                Chol_eps=np.eye(n) * np.sqrt(alpha) * sigma,
                model=model,
                rand_type=rand_type,
                use_expectation=use_expectation,
                est_risk=est_risk,
            )

            G = fitted_model.get_group_X(X)
            y_fit = [y if rand_type == "full" else w[:, i] for i in np.arange(len(G))]
            # if rand_type == 'full':
            # 	y_fit = [y for _ in np.arange(len(G))]
            # else:
            # 	y_fit = [w[:,i] for i in np.arange(len(G))]
            self.test_err[i] = test_est(
                model=LinearRegression(),
                X=G,
                y=y_fit,
                y_test=y_test,
                Chol_t=np.eye(n) * sigma,
                est_risk=est_risk,
            )[0]

        return (
            self.test_err,
            # self.tree_err,
            self.blur_err,
        )

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
        rand_type="full",
        use_expectation=False,
        alpha=0.05,
        est_sigma=False,
        est_risk=True,
    ):

        self.test_err = np.zeros(niter)
        self.blur_err = np.zeros(niter)
        # self.tree_err = np.zeros(niter)

        test_est = test_set_estimator
        blur_est = blur_forest
        # tree_est = blur_linear_selector

        gen_beta = X is None or beta is None

        if not gen_beta:
            mu = X @ beta
            sigma_true = np.sqrt(np.var(mu) / snr)
            n, p = X.shape

        if Chol_t is None:
            Chol_t = np.eye(n)
        if Theta_p is not None:
            Theta_p = Theta_p / (sigma_true**2)

        for i in np.arange(niter):

            if gen_beta:
                X = np.random.randn(n, p)
                beta = np.zeros(p)
                idx = np.random.choice(p, size=s)
                beta[idx] = np.random.uniform(-1, 1, size=s)

                mu = X @ beta
                sigma_true = np.sqrt(np.var(mu) / snr)

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

                model.fit(X, y)
                P = model.get_linear_smoother(X)
                df = np.diag(P).sum()
                yhat = P @ y
                sigma = np.sum((y - yhat) ** 2) / (n - df)

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

            (self.blur_err[i], fitted_model, w) = blur_est(
                X,
                y,
                # eps=w - y,
                Chol_t=Chol_t * sigma,
                Chol_eps=Chol_t * np.sqrt(alpha) * sigma,
                Theta_p=Theta_p,
                model=model,
                rand_type=rand_type,
                use_expectation=use_expectation,
                est_risk=est_risk,
            )

            G = fitted_model.get_group_X(X)
            y_fit = [y if rand_type == "full" else w[:, i] for i in np.arange(len(G))]
            # if rand_type == 'full':
            # 	y_fit = [y for _ in np.arange(len(G))]
            # else:
            # 	y_fit = [w[:,i] for i in np.arange(len(G))]
            self.test_err[i] = test_est(
                model=LinearRegression(),
                X=G,
                y=y_fit,
                y_test=y_test,
                Chol_t=Chol_t * sigma,
                # Chol_eps=Chol_t*np.sqrt(alpha)*sigma,
                Theta_p=Theta_p,
                est_risk=est_risk,
            )[0]

        return (
            self.test_err,
            # self.tree_err,
            self.blur_err,
        )

    def compareBlurLinearIID(
        self,
        niter=100,
        n=100,
        p=20,
        s=20,
        snr=0.4,
        X=None,
        beta=None,
        model=LinearRegression(),
        alpha=0.05,
        est_risk=True,
    ):

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
            sigma = np.sqrt(np.var(mu) / snr)
            n, p = X.shape

        for i in np.arange(niter):

            if gen_beta:
                X = np.random.randn(n, p)
                beta = np.zeros(p)
                idx = np.random.choice(p, size=s)
                beta[idx] = np.random.uniform(-1, 1, size=s)

                mu = X @ beta
                sigma = np.sqrt(np.var(mu) / snr)

            y = mu + sigma * np.random.randn(n)
            y_test = mu + sigma * np.random.randn(n)
            y_alpha = mu + sigma * np.sqrt(1 + alpha) * np.random.randn(n)
            y_test_alpha = mu + sigma * np.sqrt(1 + alpha) * np.random.randn(n)

            self.test_err[i] = test_est(
                model=model,
                X=X,
                y=y,
                y_test=y_test,
                Chol_t=np.eye(n) * sigma,
                est_risk=est_risk,
            )[0]
            self.test_err_alpha[i] = test_est(
                model=model,
                X=X,
                y=y_alpha,
                y_test=y_test_alpha,
                Chol_t=np.eye(n) * np.sqrt(1 + alpha) * sigma,
                est_risk=est_risk,
            )[0]
            self.cb_err[i] = cb_est(
                X=X,
                y=y,
                Chol_t=np.eye(n) * sigma,
                Chol_eps=np.eye(n) * np.sqrt(alpha) * sigma,
                model=model,
                est_risk=est_risk,
            )[0]
            self.blur_err[i] = blur_est(
                X,
                y,
                Chol_t=np.eye(n) * sigma,
                Chol_eps=np.eye(n) * np.sqrt(alpha) * sigma,
                model=model,
                est_risk=est_risk,
            )[0]

        return (self.test_err, self.test_err_alpha, self.cb_err, self.blur_err)

    def compareBlurLinear(
        self,
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
        est_risk=True,
    ):

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
            sigma = np.sqrt(np.var(mu) / snr)
            n, p = X.shape

        # print(n*sigma**2)
        # print("------")

        if Chol_t is None:
            Chol_t = np.eye(n)
        if Theta_p is not None:
            Theta_p = Theta_p / (sigma**2)

        for i in np.arange(niter):

            if gen_beta:
                X = np.random.randn(n, p)
                beta = np.zeros(p)
                idx = np.random.choice(p, size=s)
                beta[idx] = np.random.uniform(-1, 1, size=s)

                mu = X @ beta
                sigma = np.sqrt(np.var(mu) / snr)

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

            self.test_err[i] = test_est(
                model=model,
                X=X,
                y=y,
                y_test=y_test,
                Chol_t=Chol_t * sigma,
                Theta_p=Theta_p,
                Theta_e=None,
                est_risk=est_risk,
            )[0]
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
            self.blur_err[i] = blur_est(
                X,
                y,
                Chol_t=Chol_t * sigma,
                Chol_eps=Chol_t * np.sqrt(alpha) * sigma,
                Theta_p=Theta_p,
                Theta_e=None,
                alpha=alpha,
                # Theta_e=np.linalg.inv(Chol_t @ Chol_t.T)/(alpha*sigma**2),
                model=model,
                est_risk=est_risk,
            )[0]

        return (self.test_err, self.test_err_alpha, self.cb_err, self.blur_err)
