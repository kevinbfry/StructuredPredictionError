import plotly.express as px

GENCP_COLOR = px.colors.qualitative.Bold[0]
KFCV_COLOR = px.colors.qualitative.Bold[1]
SPCV_COLOR = px.colors.qualitative.Bold[2]
BLOOCV_COLOR = px.colors.qualitative.Bold[3]
TRGENCP_COLOR = px.colors.qualitative.Bold[4]
ESTGENCP_COLOR = px.colors.qualitative.Bold[5]
BY05_COLOR = px.colors.qualitative.Bold[6]
BY1_COLOR = px.colors.qualitative.Bold[7]
BY5_COLOR = px.colors.qualitative.Bold[8]
SPLIT_COLOR = px.colors.qualitative.Bold[9]

## Model parameters
def get_model_array(models):
    return [m.__name__ for m in models]

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
FAIR = False
EST_SIGMA = False
FRIEDMAN_MU = False

## Estimator parameters
INCL_CV = True

## Plotting parameters
COLORS = [GENCP_COLOR, KFCV_COLOR, SPCV_COLOR]
EST_NAMES = ["GenCp", "KFCV", "SPCV"]
HAS_ELEV_ERR = True
