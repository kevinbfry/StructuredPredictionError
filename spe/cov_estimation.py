import numpy as np
from numpy.linalg import LinAlgError
from scipy.spatial import distance_matrix
import skgstat as skg


def est_Sigma(
    # self,
    # locs_tr,
    X,
    y,
    locs, ## TODO: only needed now to work with how _forest.py eps variable is processed. should fix and then remove this too
    # tr_idx,
    # ts_idx,
    est_sigma,
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
    # V = skg.Variogram(locs, resids, model='matern', maxlag='median')
    
    # fitted_vm = V.fitted_model
    # full_distance = distance_matrix(locs, locs)
    # semivar = fitted_vm(full_distance.flatten()).reshape((n,n))

    # K0 = V.parameters[1] ## use sill as estimate of variance
    # est_Sigma_full = K0*np.ones_like(semivar) - semivar
    # est_Chol_t = np.linalg.cholesky(est_Sigma_full)#[tr_idx,:][:,tr_idx])
    # # est_Chol_s = np.linalg.cholesky(est_Sigma_full[ts_idx,:][:,ts_idx])
    # # self.Chol_t = est_Chol_t
    # return est_Chol_t#, est_Chol_s

    V = skg.Variogram(locs, resids, model='matern', maxlag='median', use_nugget=True)
    
    fitted_vm = V.fitted_model
    full_distance = distance_matrix(locs, locs)
    semivar = fitted_vm(full_distance.flatten()).reshape((n,n))

    K0 = V.parameters[1] ## use sill as estimate of variance
    N0 = V.parameters[-1] ## use sill as estimate of variance
    sill = K0 + N0
    est_Sigma_S = sill*np.ones_like(semivar) - semivar
    est_Sigma_M = N0 * np.eye(semivar.shape[0])
    est_Sigma_F = est_Sigma_S + est_Sigma_M

    def get_cholesky(Sigma):
        try:
            chol = np.linalg.cholesky(Sigma)#[tr_idx,:][:,tr_idx])
        # except np.linalg.LinAlgError as e:
        except LinAlgError as err:#Exception as e:
            if str(err) == "Matrix is not positive definite":
                # eigv = np.linalg.eigh(Sigma)[0]
                # if eigv[1] > 0:
                #     eigv = -eigv[0]
                #     chol = np.linalg.cholesky(Sigma + eigv*np.eye(Sigma.shape[0]))
                # else:
                #     raise ValueError("At least two eigenvalues of 'est_Sigma_F' negative")

                ## instead of doing eigendecomp, just add some proportion of trace
                c = 1e-6/n
                trc = np.sum(np.diag(Sigma))
                chol = np.linalg.cholesky(Sigma + c*trc*np.eye(n))
            else:
                raise

        return chol
    
    est_Chol_F = get_cholesky(est_Sigma_F)
    est_Chol_S = get_cholesky(est_Sigma_S)

    # try:
    #     est_Chol_F = np.linalg.cholesky(est_Sigma_F)#[tr_idx,:][:,tr_idx])
    # except np.linalg.LinAlgError as e:
    #     if str(e) == "Matrix is not positive definite":
    #         eigv = np.linalg.eigh(est_Sigma_F)[0]
    #         if eigv[1] > 0:
    #             eigv = -eigv[0]
    #             est_Chol_F = np.linalg.cholesky(est_Sigma_F + eigv*np.eye(est_Sigma_F.shape[0]))
    #         else:
    #             raise ValueError("At least two eigenvalues of 'est_Sigma_F' negative")
    #     raise
    
    # # est_Chol_S = np.linalg.cholesky(K0*np.ones_like(semivar) - semivar)
    # try:
    #     est_Chol_S = np.linalg.cholesky(est_Sigma_S)#[tr_idx,:][:,tr_idx])
    # except np.linalg.LinAlgError as e:
    #     if str(e) == "Matrix is not positive definite":
    #         eigv = np.linalg.eigh(est_Sigma_S)[0]
    #         if eigv[1] > 0:
    #             eigv = -eigv[0]
    #             est_Chol_S = np.linalg.cholesky(est_Sigma_S + eigv*np.eye(est_Sigma_S.shape[0]))
    #         else:
    #             raise ValueError("At least two eigenvalues of 'est_Sigma_F' negative")
    #     raise

    if est_sigma == 'corr_resp':
        return est_Chol_F, est_Chol_S
    
    return est_Chol_F
    '''
    for nugget:
    Nugget is measurement error/microscale variation.
    structured part is like above but use partial sill (K0 - nugget)
    rather than sill (K0).
    '''