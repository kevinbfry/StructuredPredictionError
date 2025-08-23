from docs.sim_parameters.sims_base_config import *
from spe.estimators import cp_smoother as spe_est
from spe.smoothers import LinearRegression

## data generation parameters
P = 5
S = 5
NOISE_LENGTH_SCALE = 1.

## Model parameters
MODEL_NAMES = get_model_array([
    LinearRegression
])

MODEL_KWARGS = [
    {'fit_intercept': False}
]

## Estimator parameters
SPE_EST_STR = spe_est.__name__
SPE_EST_STR_LIST = [SPE_EST_STR]
FIG_NAME_PREFIX = "Linear"

TWO_Y_EST = False

nboot = 100
k = 5
EST_KWARGS= [
    {},
    {},
    {'k': k},
    {'k': k}
]

## Plotting parameters
HAS_ELEV_ERR = False