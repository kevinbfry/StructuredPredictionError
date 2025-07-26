import numpy as np
from sklearn.gaussian_process.kernels import RBF, Matern


def gen_tr_idxs(n, tr_frac, use_spatial_split=False):
    if tr_frac == 1:
        return np.ones(n, dtype=bool)
    if use_spatial_split:
        return create_clus_split(
            int(np.sqrt(n)), int(np.sqrt(n)), tr_frac
        )
    else:
        tr_idx = np.concatenate((np.ones(int(tr_frac*n)), np.zeros(n - int(tr_frac*n)))).astype(bool)
        np.random.shuffle(tr_idx)
        return tr_idx


def gen_coords(sqrt_n, gsize):
    nx = ny = sqrt_n#int(np.sqrt(n))
    xs = np.linspace(0, gsize, nx)
    ys = np.linspace(0, gsize, ny)
    c_x, c_y = np.meshgrid(xs, ys)
    c_x = c_x.flatten()
    c_y = c_y.flatten()
    coord = np.stack([c_x, c_y]).T

    return coord


def gen_matern_X(
    c_x,
    c_y,
    p,
    length_scale=5,
    nu=1.5,
    nspikes=None,
):
    n = len(c_x)
    matern_kernel = Matern(length_scale=length_scale, nu=nu)
    cov = gen_cov_mat(c_x, c_y, matern_kernel)

    return _gen_X(n, p, cov, nspikes)


def gen_rbf_X(
    c_x,
    c_y,
    p,
    length_scale=5,
    nspikes=None,
):
    n = len(c_x)

    rbf_kernel = RBF(length_scale)
    cov = gen_cov_mat(c_x, c_y, rbf_kernel)

    return _gen_X(n, p, cov, nspikes)


def gen_cov_mat(
    c_x,
    c_y,
    kernel,
):
    if c_y is not None:
        locs = np.stack([c_x, c_y]).T
    else:
        locs = np.atleast_2d(c_x).T
    return kernel(locs)


def gen_chol_cov(delta, c_x, c_y, noise_kernel, noise_length_scale, noise_nu=None):
    n = len(c_x)
    if noise_kernel == 'rbf':
        Sigma_t = gen_cov_mat(c_x, c_y, RBF(length_scale=noise_length_scale))
    elif noise_kernel == 'matern':
        Sigma_t = gen_cov_mat(c_x, c_y, Matern(length_scale=noise_length_scale, nu=noise_nu))
    else:
        Sigma_t = np.eye(n)
    
    Cov_y_ystar = delta*Sigma_t
    Sigma_t = delta*Sigma_t + (1-delta)*np.eye(n)

    if noise_kernel == 'rbf' or noise_kernel == 'matern':
        Chol_y = np.linalg.cholesky(Sigma_t)
    else:
        Chol_y = np.eye(n)

    return Chol_y, Cov_y_ystar


def _gen_X(n, p, cov, nspikes=None):
    if nspikes is None:
        nspikes = int(2 * np.log2(n))

    X = np.zeros((n, p))
    for i in np.arange(p):
        spikes = np.random.uniform(1, 3, size=nspikes) * np.random.choice(
            [-1, 1], size=nspikes, replace=True
        )
        spike_idx = np.random.choice(n, size=nspikes, replace=False)

        W = cov[:, spike_idx]
        W = W / W.sum(axis=1)[:, None]
        X[:, i] = W @ spikes

    return X


### this samples grid, then uniformly within grid.
def create_clus_split(
    nx,
    ny,
    tr_frac,
    ngrid=5,
    ts_frac=None,
    sort_grids=False,
    return_groups=False,
):
    ngrid = int(np.amin([nx/2, ny/2, ngrid])) ## a grid square must have at least 4 points

    xv, yv = np.meshgrid(np.arange(nx), np.arange(ny))
    pts = np.stack([xv.ravel(), yv.ravel()]).T
    n = nx * ny

    cxv, cyv = np.meshgrid(np.arange(ngrid), np.arange(ngrid))
    cpts = np.stack([cxv.ravel(), cyv.ravel()]).T

    incrx = int(nx / ngrid)
    incry = int(ny / ngrid)
    nsq = ngrid**2
    grids = []
    for cpt in cpts:
        gxv, gyv = np.meshgrid(
            cpt[0] * incrx + np.arange(incrx), cpt[1] * incry + np.arange(incry)
        )
        grids.append([(x, y) for (x, y) in zip(gxv.ravel(), gyv.ravel())])
    grids = np.array(grids)


    n_tr_centers = (
        int(tr_frac * nsq) if tr_frac > .5 
        else np.minimum(nsq, int(tr_frac * nsq * 2))
    )
    if ts_frac is not None:
        n_ts_centers = (
            int(ts_frac * nsq) if tr_frac > .5 
            else np.minimum(nsq, int(ts_frac * nsq * 2))
        )
    else:
        ts_frac = 0
        n_ts_centers = 0

    n_centers = n_tr_centers + n_ts_centers

    grid_idx = np.random.choice(nsq, size=n_centers, replace=False)

    if sort_grids:
        grid_idx = np.sort(grid_idx)

    sel_grids = grids[grid_idx[:n_tr_centers]]
    sel_ts_grids = grids[grid_idx[n_tr_centers:(n_tr_centers+1)]]#:]]

    groups = np.zeros(n)

    tr_idx = []
    sample_frac = n * (tr_frac) / (n_tr_centers * incrx * incry)
    for i, g in enumerate(sel_grids):
        group_i = g[np.random.choice(len(g), size=int(len(g) * sample_frac), replace=False)]
        tr_idx.append(
            group_i
        )
        groups[np.ravel_multi_index(np.array(group_i).T, (nx, ny))] = i+1

    tr_idx = np.vstack(tr_idx)

    tr_idx = np.ravel_multi_index(tr_idx.T, (nx, ny))

    tr_bool = np.zeros(n).astype(bool)
    tr_bool[tr_idx] = True

    if ts_frac > 0:
        ts_idx = []
        sample_ts_frac = n * (ts_frac) / (n_ts_centers * incrx * incry)
        for g in sel_ts_grids:
            ts_idx.append(
                g[np.random.choice(len(g), size=int(len(g) * sample_ts_frac), replace=False)]
            )

        ts_idx = np.vstack(ts_idx)

        ts_idx = np.ravel_multi_index(ts_idx.T, (nx, ny))

        ts_bool = np.zeros(n).astype(bool)
        ts_bool[ts_idx] = True
        
        assert(np.amax(tr_bool + ts_bool) == 1)

        if return_groups:
            return tr_idx, ts_idx, groups

        return tr_idx, ts_idx

    return tr_bool
