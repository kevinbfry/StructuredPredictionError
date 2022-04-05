import numpy as np
from spe.mse_estimator import MSESimulator

def test_relaxed_lasso(n=200,
                       p=30,
                       reps=1,
                       train_frac=.5,
                       test_frac=.5,
                       niter=5,
                       eps_sigma=np.sqrt(10),
                       block_corr=0.,
                       inter_corr=0.,
                       fit_intercept=True,
                       pred_type='test',
                       model_type='lasso',
                       lambd=0.5): 

    mse_sim = MSESimulator()

    (true_mse_tst,
     kfcv_mse_tst, 
     spcv_mse_tst,
     gmcp_mse_tst,
     frft_mse_tst,
     nhnst_mse_tst, 
     hnst_mse_tst) = mse_sim.cv_compare(niter, 
                                        n=n,
                                        p=p,
                                        reps=reps,
                                        train_frac=train_frac, 
                                        test_frac=test_frac,
                                        eps_sigma=eps_sigma,
                                        block_corr=block_corr,
                                        inter_corr=inter_corr,
                                        fit_intercept=fit_intercept,
                                        pred_type=pred_type,
                                        model_type=model_type,
                                        lambd=lambd)

