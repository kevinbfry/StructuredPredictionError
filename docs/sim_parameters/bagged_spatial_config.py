import plotly.express as px

from docs.sim_parameters.sims_base_config import *
from spe.tree import Tree
from spe.relaxed_lasso import RelaxedLasso
from spe.estimators import new_y_est, simple_train_test_split, cp_bagged as spe_est

## data generation parameters
SQRT_N = 20
P = 5
TR_FRAC = .25
USE_SPATIAL_SPLIT = True

NOISE_LENGTH_SCALE = 1.#5.

## Model parameters
MODEL_NAMES = get_model_array([
    RelaxedLasso,
    Tree,
])

lambd = .31
max_depth = 2
MODEL_KWARGS = [
    {'lambd': lambd},
    {'max_depth': max_depth, 
     'max_features': 'sqrt'}
]

## Estimator parameters
SPE_EST_STR = spe_est.__name__
FIG_NAME_PREFIX = SPE_EST_STR + "_spatial"
FIG_NAME_PREFIX = FIG_NAME_PREFIX.replace("_", " ").title()

EST_STRS = [
    new_y_est.__name__,
    spe_est.__name__,
    simple_train_test_split.__name__,
]

full_refit = False
EST_KWARGS = [
    {'alpha': None,
    'full_refit': full_refit,
    'bagg': True},
    {'full_refit': full_refit},
    {},
]

## ErrorComparer parameters
FRIEDMAN_MU = True

## Plotting parameters
COLORS=[GENCP_COLOR, SPLIT_COLOR]
EST_NAMES = ["GenCp", "Split"]
HAS_ELEV_ERR = False

## Markdown parameters
MARKDOWN_STR = "# Bagged Models\n Here we demonstrate the effectiveness of ```{est_md_str}``` to estimate spatial split MSE on simulated data."