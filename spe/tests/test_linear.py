import numpy as np
from spe.mse_estimator import ErrorComparer

def test_linear(n=100): 
    nx = ny = int(np.sqrt(n))
    xs = np.linspace(0, 10, nx)
    ys = np.linspace(0, 10, ny)
    c_x, c_y = np.meshgrid(xs, ys)
    c_x = c_x.flatten()
    c_y = c_y.flatten()
    coord = np.stack([c_x, c_y]).T

    mse_sim = ErrorComparer()

    (true_mse_tst,
     kfcv_mse_tst, 
     spcv_mse_tst,
     gmcp_mse_tst) = mse_sim.compareLinearTrTs(
                                        niter=10,
                                        n=n,
                                        p=30,
                                        s=5,
                                        snr=0.4, 
                                        X=None,
                                        beta=None,
                                        coord=coord,
                                        Chol_t=None,
                                        Chol_s=None,
                                        tr_idx=None,
                                        k=10,
                                        )

