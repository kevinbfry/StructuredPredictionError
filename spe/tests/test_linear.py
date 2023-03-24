import numpy as np
from sklearn.linear_model import LinearRegression

from spe.estimators import cp_linear_train_test

## simple simple execution test
def test_lin(
    niter=1,
    n=200,
    p=400,
):
    
    for i in np.arange(niter):
        X = np.random.randn((n,p))
        y = np.random.randn(n)
        tr_idx = np.ones(n)

        cp_linear_train_test(
            LinearRegression(intercept=False),
            X,
            y,
            tr_idx,
            Chol_t=None,
            Chol_s=None,
        )