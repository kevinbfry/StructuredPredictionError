import numpy as np
from sklearn.gaussian_process.kernels import RBF, Matern


def gen_matern_X(
    c_x,
    c_y,
    p,
    length_scale=5,
    nu=1.5,
    nspikes=None,
):
    # locs = np.stack([c_x, c_y]).T

    # n = len(locs)
    n = len(c_x)

    matern_kernel = Matern(length_scale=length_scale, nu=nu)
    # cov = matern_kernel(locs)
    cov = gen_cov_mat(c_x, c_y, matern_kernel)

    return _gen_X(n, p, cov, nspikes)


def gen_rbf_X(
    c_x,
    c_y,
    p,
    length_scale=5,
    nspikes=None,
):
    # locs = np.stack([c_x, c_y]).T

    # n = len(locs)
    n = len(c_x)

    rbf_kernel = RBF(length_scale)
    # cov = rbf_kernel(locs)
    cov = gen_cov_mat(c_x, c_y, rbf_kernel)

    return _gen_X(n, p, cov, nspikes)


def gen_cov_mat(
    c_x,
    c_y,
    kernel,
):
    locs = np.stack([c_x, c_y]).T
    return kernel(locs)


def _gen_X(n, p, cov, nspikes=None):
    if nspikes is None:
        nspikes = int(2 * np.log2(n))
        # nspikes = int(np.log2(n))

    X = np.zeros((n, p))
    for i in np.arange(p):
        spikes = np.random.uniform(1, 3, size=nspikes) * np.random.choice(
            [-1, 1], size=nspikes, replace=True
        )
        spike_idx = np.random.choice(n, size=nspikes, replace=False)

        W = cov[:, spike_idx]
        W = W / W.sum(axis=1)[:, None]
        X[:, i] = W @ spikes

        # X[spike_idx,i] = spikes

        # spike_bool = np.zeros(n, dtype=bool)
        # spike_bool[spike_idx] = True
        # nonspike_bool = np.ones(n) - spike_bool
        # nonspike_bool = nonspike_bool.astype(bool)

        # S12 = cov[nonspike_bool,:][:,spike_bool]
        # S22 = cov[spike_bool,:][:,spike_bool]
        # W = S12 @ np.linalg.inv(S22)
        # X[nonspike_bool,i] = W @ spikes

    return X


# def create_clus_split(
# 	nx,
# 	ny,
# 	tr_frac,
# 	n_centers=15,
# 	):
# 	xv, yv = np.meshgrid(np.arange(nx), np.arange(ny))
# 	pts = np.stack([xv.ravel(), yv.ravel()]).T
# 	n = nx*ny

# 	if n_centers is None or n_centers == 'log':
# 		n_centers = 2*int(np.log2(n))
# 	elif n_centers == 'sqrt':
# 		n_centers = int(np.sqrt(n))

# 	ctr = np.random.choice(pts.shape[0], size=n_centers, replace=True)
# 	ctr = pts[ctr]
# 	# print("centers")
# 	# print(ctr)
# 	# print("------------------------")

# 	tr_idx = np.vstack([[pt + np.array((1.25*np.random.randn(2)).astype(int)) for _ in np.arange(int(3.5*n/n_centers))] for pt in ctr])

# 	# n_sample = int(tr_frac*n)
# 	# tr_idx = np.zeros((n_sample,2), dtype=int)
# 	# tr_idx[:n_centers,:] = ctr
# 	# for i in np.arange(n_centers, n_sample):
# 	# 	j = int(i % n_centers)
# 	# 	while True:
# 	# 		prop_pt = pts[j] + np.array(2.5*np.random.randn(2)).astype(int)
# 	# 		# print(prop_pt)
# 	# 		if prop_pt.tolist() not in tr_idx.tolist():
# 	# 			tr_idx[i,:] = prop_pt
# 	# 			break

# 	tr_idx = np.maximum(0, tr_idx)
# 	tr_idx[:,0] = cx = np.minimum(nx-1, tr_idx[:,0])
# 	tr_idx[:,1] = cy = np.minimum(ny-1, tr_idx[:,1])
# 	tr_idx = np.unique(np.ravel_multi_index(tr_idx.T, (nx,ny)))

# 	tr_bool = np.zeros(n).astype(bool)
# 	tr_bool[tr_idx] = True

# 	return tr_bool


### this samples grid, then uniform within grid.
def create_clus_split(
    nx,
    ny,
    tr_frac,
    ngrid=5,
):
    xv, yv = np.meshgrid(np.arange(nx), np.arange(ny))
    pts = np.stack([xv.ravel(), yv.ravel()]).T
    n = nx * ny

    cxv, cyv = np.meshgrid(np.arange(ngrid), np.arange(ngrid))
    cpts = np.stack([cxv.ravel(), cyv.ravel()]).T
    # selected_grids = cpts[ctr_idx,:]

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

    n_centers = np.minimum(nsq, int(tr_frac * nsq * 2))
    grid_idx = np.random.choice(nsq, size=n_centers, replace=False)
    sel_grids = grids[grid_idx]

    tr_idx = []
    sample_frac = n * tr_frac / (n_centers * incrx * incry)
    for g in sel_grids:
        tr_idx.append(
            g[np.random.choice(len(g), size=int(len(g) * sample_frac), replace=False)]
        )

    tr_idx = np.vstack(tr_idx)

    tr_idx = np.ravel_multi_index(tr_idx.T, (nx, ny))

    tr_bool = np.zeros(n).astype(bool)
    tr_bool[tr_idx] = True

    return tr_bool
