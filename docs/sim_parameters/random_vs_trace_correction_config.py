import numpy as np
from sklearn.linear_model import LassoCV, RidgeCV

from docs.sim_parameters.sims_base_config import *
from spe.estimators import cp_arbitrary as spe_est

## data generation parameters
SQRT_N = 20#**2

## Model parameters
MODEL_NAMES = get_model_array([
    LassoCV, RidgeCV
])

alphas = np.logspace(.01, 10, 5).tolist()
MODEL_KWARGS = [
    {'alphas': alphas},
    {'alphas': alphas},
]

## Estimator parameters
SPE_EST_STR = spe_est.__name__
FIG_NAME_PREFIX = "Randomized vs Trace Correction"
SPE_EST_STR_LIST = [SPE_EST_STR]*2
INCL_CV = False

alpha = .05
nboot = 100
EST_KWARGS = [
    {'alpha': None,
    'full_refit': False},
    {'alpha': alpha,
    'full_refit': False},
    {'alpha': alpha, 
    'use_trace_corr': False, 
    'nboot': nboot},
    {'alpha': alpha, 
    'use_trace_corr': True, 
    'nboot': nboot}
]

## Plotting parameters
COLORS = [GENCP_COLOR, TRGENCP_COLOR]
EST_NAMES = ["Rand Corr", "Trace Corr"]
ERR_BARS = True

## Markdown parameters
LATEX_STR = "$\mathrm{tr}(\Theta_p (\Sigma_{Y^*} - \Sigma_Y)) - \|\Sigma_Y\Sigma_\omega^{-1}\omega\|_2^2$ compared with the deterministic trace correction $\mathrm{tr}(\Theta_p (\Sigma_{Y^*} - \Sigma_{W^\perp}))$."
MARKDOWN_STR = "# Variance Reduction: Randomized vs Trace Correction\n Simulations demonstrating the variance reduction in using the randomized correction {latex_str}"

