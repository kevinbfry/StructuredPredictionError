import plotly.express as px

from docs.sim_parameters.sims_base_config import *
from spe.tree import Tree
from spe.estimators import cp_arbitrary as spe_est

## data generation parameters
SQRT_N = 20
P = 5
TR_FRAC = .25
USE_SPATIAL_SPLIT = False

## Model parameters
MODEL_NAMES = get_model_array([
    Tree,
])

max_depth = 2
MODEL_KWARGS = [
    {'max_depth': max_depth}
]

## Estimator parameters
SPE_EST_STR = spe_est.__name__
SPE_EST_STR_LIST = [SPE_EST_STR]*2

alpha = .05
nboot = 100
k = 5
EST_KWARGS = [
    {'alpha': None},
    {'alpha': alpha},
    {'alpha': alpha,
     'nboot': nboot},
    {'alpha': alpha,
     'nboot': nboot},
    {'k': k},
    {'k': k}
]

## ErrorComparer parameters
EST_SIGMA = [False, False, False, True, False ,False]

## Plotting parameters
COLORS=[GENCP_COLOR, ESTGENCP_COLOR, KFCV_COLOR, SPCV_COLOR]
EST_NAMES = ["OGenCp", "EGenCp", "KFCV", "SPCV"]

## Markdown parameters
MARKDOWN_STR = "# Estimated vs Oracle Covariance\n Here we demonstrate the effectiveness of estimating the covariance matrix on the quality of estimates from ```{est_md_str}``` on simulated data."