import numpy as np
from numpy.linalg import LinAlgError
from scipy.spatial import distance_matrix
import skgstat as skg


def est_Sigma(
    X,
    y,
    locs, ## TODO: only needed now to work with how _forest.py eps variable is processed. should fix and then remove this too
    est_sigma,
    est_sigma_model, 
):
    n = locs.shape[0]

    if est_sigma_model is None:
        raise ValueError("Must provide est_simga_model")

    est_sigma_model.fit(X, y)
    resids =  y - est_sigma_model.predict(X)

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
        except LinAlgError as err:
            if str(err) == "Matrix is not positive definite":
                ## instead of doing eigendecomp, just add some proportion of trace
                c = 1e-6/n
                trc = np.sum(np.diag(Sigma))
                chol = np.linalg.cholesky(Sigma + c*trc*np.eye(n))
            else:
                raise

        return chol
    
    est_Chol_F = get_cholesky(est_Sigma_F)
    est_Chol_S = get_cholesky(est_Sigma_S)

    if est_sigma == 'corr_resp':
        return est_Chol_F, est_Chol_S
    
    return est_Chol_F