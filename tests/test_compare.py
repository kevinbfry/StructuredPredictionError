import numpy as np
from sklearn.linear_model import LinearRegression

from spe.relaxed_lasso import BaggedRelaxedLasso, RelaxedLasso
from spe.forest import BlurredForest
from .mse_estimator import ErrorComparer
from spe.estimators import (
    kfoldcv,
    kmeanscv,
    cb,
    cb_isotropic,
    blur_linear,
    blur_linear_selector,
    blur_forest,
    cp_linear_train_test,
    cp_relaxed_lasso_train_test,
    cp_bagged_train_test,
    cp_rf_train_test,
    better_test_est_split,
    bag_kfoldcv,
    bag_kmeanscv,
)


def test_lin(n=100):
    nx = ny = int(np.sqrt(n))
    xs = np.linspace(0, 10, nx)
    ys = np.linspace(0, 10, ny)
    c_x, c_y = np.meshgrid(xs, ys)
    c_x = c_x.flatten()
    c_y = c_y.flatten()
    coord = np.stack([c_x, c_y]).T

    mse_sim = ErrorComparer()

    (true_mse_tst, kfcv_mse_tst, spcv_mse_tst, gmcp_mse_tst) = mse_sim.compare(
        LinearRegression(fit_intercept=False),
        [kfoldcv, kmeanscv, cp_linear_train_test],
        [{"k": 5}, {"k": 5}, {}],
        niter=10,
        n=n,
        p=30,
        s=30,
        snr=0.4,
        X=None,
        beta=None,
        coord=coord,
        Chol_t=None,
        Chol_s=None,
        tr_idx=None,
        fair=False,
    )


def test_relaxed_lasso(n=100):
    nx = ny = int(np.sqrt(n))
    xs = np.linspace(0, 10, nx)
    ys = np.linspace(0, 10, ny)
    c_x, c_y = np.meshgrid(xs, ys)
    c_x = c_x.flatten()
    c_y = c_y.flatten()
    coord = np.stack([c_x, c_y]).T

    mse_sim = ErrorComparer()

    (true_mse_tst, kfcv_mse_tst, spcv_mse_tst, gmcp_mse_tst) = mse_sim.compare(
        RelaxedLasso(lambd=0.1),
        [kfoldcv, kmeanscv, cp_relaxed_lasso_train_test],
        [{"k": 5}, {"k": 5}, {"alpha": 0.5, "use_trace_corr": False}],
        niter=10,
        n=n,
        p=30,
        s=30,
        snr=0.4,
        X=None,
        beta=None,
        coord=coord,
        Chol_t=None,
        Chol_s=None,
        tr_idx=None,
        fair=False,
    )


def test_relaxed_lasso_fair(n=100):
    nx = ny = int(np.sqrt(n))
    xs = np.linspace(0, 10, nx)
    ys = np.linspace(0, 10, ny)
    c_x, c_y = np.meshgrid(xs, ys)
    c_x = c_x.flatten()
    c_y = c_y.flatten()
    coord = np.stack([c_x, c_y]).T

    mse_sim = ErrorComparer()

    (true_mse_tst, kfcv_mse_tst, spcv_mse_tst, gmcp_mse_tst) = mse_sim.compare(
        RelaxedLasso(lambd=0.1),
        [kfoldcv, kmeanscv, cp_relaxed_lasso_train_test],
        [{"k": 5}, {"k": 5}, {"alpha": 0.5, "use_trace_corr": False}],
        niter=10,
        n=n,
        p=30,
        s=30,
        snr=0.4,
        X=None,
        beta=None,
        coord=coord,
        Chol_t=None,
        Chol_s=None,
        tr_idx=None,
        fair=True,
    )


def test_bagged_relaxed_lasso(n=100):
    nx = ny = int(np.sqrt(n))
    xs = np.linspace(0, 10, nx)
    ys = np.linspace(0, 10, ny)
    c_x, c_y = np.meshgrid(xs, ys)
    c_x = c_x.flatten()
    c_y = c_y.flatten()
    coord = np.stack([c_x, c_y]).T

    mse_sim = ErrorComparer()

    (true_mse_tst, kfcv_mse_tst, spcv_mse_tst, gmcp_mse_tst) = mse_sim.compare(
        BaggedRelaxedLasso(base_estimator=RelaxedLasso(lambd=0.5), n_estimators=5),
        [bag_kfoldcv, bag_kmeanscv, cp_bagged_train_test],
        [{"k": 5}, {"k": 5}, {"use_trace_corr": False}],
        niter=10,
        n=n,
        p=30,
        s=30,
        snr=0.4,
        X=None,
        beta=None,
        coord=coord,
        Chol_t=None,
        Chol_s=None,
        tr_idx=None,
        fair=False,
    )


def test_blurred_forest(n=100):
    nx = ny = int(np.sqrt(n))
    xs = np.linspace(0, 10, nx)
    ys = np.linspace(0, 10, ny)
    c_x, c_y = np.meshgrid(xs, ys)
    c_x = c_x.flatten()
    c_y = c_y.flatten()
    coord = np.stack([c_x, c_y]).T

    mse_sim = ErrorComparer()

    (true_mse_tst, kfcv_mse_tst, spcv_mse_tst, gmcp_mse_tst) = mse_sim.compare(
        BlurredForest(n_estimators=5),
        [bag_kfoldcv, bag_kmeanscv, cp_rf_train_test],
        [{"k": 5}, {"k": 5}, {"use_trace_corr": False}],
        niter=10,
        n=n,
        p=30,
        s=30,
        snr=0.4,
        X=None,
        beta=None,
        coord=coord,
        Chol_t=None,
        Chol_s=None,
        tr_idx=None,
        fair=False,
        bootstrap_type="blur",
    )


def test_blurred_forest_fair(n=100):
    nx = ny = int(np.sqrt(n))
    xs = np.linspace(0, 10, nx)
    ys = np.linspace(0, 10, ny)
    c_x, c_y = np.meshgrid(xs, ys)
    c_x = c_x.flatten()
    c_y = c_y.flatten()
    coord = np.stack([c_x, c_y]).T

    mse_sim = ErrorComparer()

    (true_mse_tst, kfcv_mse_tst, spcv_mse_tst, gmcp_mse_tst) = mse_sim.compare(
        BlurredForest(n_estimators=5),
        [bag_kfoldcv, bag_kmeanscv, cp_rf_train_test],
        [{"k": 5}, {"k": 5}, {"use_trace_corr": False}],
        niter=10,
        n=n,
        p=30,
        s=30,
        snr=0.4,
        X=None,
        beta=None,
        coord=coord,
        Chol_t=None,
        Chol_s=None,
        tr_idx=None,
        fair=True,
        bootstrap_type="blur",
    )
