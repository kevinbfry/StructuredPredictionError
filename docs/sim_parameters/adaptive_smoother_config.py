import numpy as np
from sklearn.linear_model import LassoCV, RidgeCV, Lasso

from docs.sim_parameters.sims_base_config import *
from spe.estimators import new_y_est, kfoldcv, kmeanscv, cp_adaptive_smoother 
from spe.tree import Tree
from spe.relaxed_lasso import RelaxedLasso

spe_est = cp_adaptive_smoother

## Model parameters
def get_model_array(models):
    return [m.__name__ for m in models]

MODEL_NAMES = get_model_array([
    RelaxedLasso,
    Tree,
])
# MODEL_NAMES = ["Lasso", "Ridge CV", "Lasso CV"]

max_depth = 3
lambd = .31
MODEL_KWARGS = [
    {'lambd': lambd},
    {'max_depth': max_depth},
]

SPE_EST_STR = spe_est.__name__
## Estimator parameters
def get_est_array(est):
    return [
        new_y_est.__name__,
        new_y_est.__name__,
        est,
        kfoldcv.__name__, 
        kmeanscv.__name__
    ]

EST_STRS = get_est_array(SPE_EST_STR)

alpha = .05
nboot = 100
k = 5
EST_KWARGS= [
    {'alpha': None,
    'full_refit': False},
    {'alpha': alpha,
    'full_refit': False},
    {'alpha': alpha, 
    'use_trace_corr': False, 
    'full_refit': False,
    'nboot': nboot},
    {'k': k},
    {'k': k}
]

## Markdown parameters
MODEL_MD_STR = get_model_md_str(SPE_EST_STR)
EST_MD_STR = get_est_md_str(SPE_EST_STR)