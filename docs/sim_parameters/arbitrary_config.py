import numpy as np
from sklearn.linear_model import LassoCV, RidgeCV, Lasso

from docs.sim_parameters.sims_base_config import *
from spe.estimators import cp_arbitrary as spe_est

## Model parameters
MODEL_NAMES = get_model_array([
    Lasso,
    RidgeCV,
    LassoCV,
])

lambdas = np.logspace(.01, 10, 10).tolist()
MODEL_KWARGS = [
    {'alpha': .31},
    {'alphas': lambdas},
    {'alphas': lambdas},
]

## Estimator parameters
SPE_EST_STR = spe_est.__name__
SPE_EST_STR_LIST = [SPE_EST_STR]

alpha = .05
nboot = 100
k = 5
EST_KWARGS= [
    {'alpha': None},
    {'alpha': alpha},
    {'alpha': alpha, 
     'nboot': nboot},
    {'k': k},
    {'k': k}
]
