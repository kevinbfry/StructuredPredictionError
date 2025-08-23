from docs.sim_parameters.sims_base_config import *
from spe.estimators import cp_adaptive_smoother as spe_est
from spe.tree import Tree
from spe.relaxed_lasso import RelaxedLasso

## Model parameters
MODEL_NAMES = get_model_array([
    RelaxedLasso,
    Tree,
])

max_depth = 3
lambd = .31
MODEL_KWARGS = [
    {'lambd': lambd},
    {'max_depth': max_depth},
]

SPE_EST_STR = spe_est.__name__
SPE_EST_STR_LIST = [SPE_EST_STR]
## Estimator parameters

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