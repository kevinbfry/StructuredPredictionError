import plotly.express as px

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

## Plotting parameters
COLORS = px.colors.qualitative.Bold
EST_NAMES = ["GenCp", "KFCV", "SPCV"]
