import papermill as pm
from importlib import import_module

from spe.estimators import new_y_est, kfoldcv, kmeanscv
from sims_base_config import *


def load_strategy_module(strategy_name: str):
    """
    Dynamically imports a module based on its string name.
    Assumes strategy modules are in a 'strategies' directory/package.
    """
    try:
        # Example: strategies/email_strategy.py, strategies/sms_strategy.py
        module_path = f"strategies.{strategy_name.lower()}_strategy"
        strategy_module = import_module(module_path)
        print(f"Successfully loaded module: {module_path}")
        return strategy_module
    except ImportError:
        print(f"Error: Strategy module '{strategy_name}' not found.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None




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


pm.execute_notebook(
    './sphinx/notebooks/sim_template.ipynb',
    './sphinx/notebooks/sim_template_test_output.ipynb',
    kernel_name='spe',
    parameters=dict(
        niter=NITER,
        gsize=GSIZE,
        sqrt_n=SQRT_N,
        p=P,
        s=S,
        delta=DELTA,
        snr=SNR,
        tr_frac=TR_FRAC,
        use_spatial_split=USE_SPATIAL_SPLIT,
        noise_kernel=NOISE_KERNEL,
        noise_length_scale=NOISE_LENGTH_SCALE,
        noise_nu=NOISE_NU,
        X_kernel=X_KERNEL,
        X_length_scale=X_LENGTH_SCALE,
        X_nu=X_NU,
        alpha=ALPHA,
        nboot=NBOOT,
        Chol_ystar=CHOL_YSTAR,
        Cov_y_ystar=COV_Y_YSTAR,
        k=K,
        model_kwargs=MODEL_KWARGS,
        model_names=MODEL_NAMES,
        est_name=EST_NAMES,
        est_strs=EST_STRS,
        est_kwargs=EST_KWARGS,
        est_names=EST_NAMES,
    )
)