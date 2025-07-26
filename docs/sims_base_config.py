import numpy as np
from sklearn.linear_model import LassoCV, RidgeCV, Lasso

from spe.estimators import cp_arbitrary as spe_est


## number of realizations to run
NITER = 100

## data generation parameters
GSIZE = 100#10
SQRT_N = 10#**2
P = 200
S = 5
DELTA = 0.75
SNR = 0.4
TR_FRAC = .5
USE_SPATIAL_SPLIT = False

NOISE_KERNEL = 'matern'
NOISE_LENGTH_SCALE = 5.#1.
NOISE_NU = .5

X_KERNEL = 'matern'
X_LENGTH_SCALE = 5.
X_NU = 2.5

## ErrorComparer parameters
ALPHA = .05
NBOOT = 100
CHOL_YSTAR = None
COV_Y_YSTAR = None


## Model parameters
K = 5

def get_model_array(models):
    return [m.__name__ for m in models]

MODEL_NAMES = get_model_array([
    Lasso,
    RidgeCV,
    LassoCV,
])
# MODEL_NAMES = ["Lasso", "Ridge CV", "Lasso CV"]

lambdas = np.logspace(.01, 10, 10).tolist()
MODEL_KWARGS = [
    {'alpha': .31},
    {'alphas': lambdas},
    {'alphas': lambdas},
]

## Markdown parameters

SPE_EST_STR = spe_est.__name__
## Estimator parameters

EST_KWARGS= [
    {'alpha': None,
    'full_refit': False},
    {'alpha': ALPHA,
    'full_refit': False},
    {'alpha': ALPHA, 
    'use_trace_corr': False, 
    'nboot': NBOOT},
    {'k': K},
    {'k': K}
]
EST_NAMES = ["GenCp", "KFCV", "SPCV"]