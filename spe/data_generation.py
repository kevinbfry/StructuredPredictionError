import numpy as np
from sklearn.gaussian_process.kernels import RBF


def gen_rbf_X(c_x, c_y, p, nspikes=None):
	locs = np.stack([c_x, c_y]).T

	n = len(locs)

	rbf_kernel = RBF(5)
	cov = rbf_kernel(locs)

	if nspikes is None:
		nspikes = int(2*np.log2(n))

	X = np.zeros((n, p))
	for i in np.arange(p):
		spikes = np.random.uniform(1,3, size=nspikes) * np.random.choice([-1,1],size=nspikes,replace=True)
		spike_idx = np.random.choice(n, size=nspikes)

		W = cov[:, spike_idx]
		W = W / W.sum(axis=1)[:,None]
		X[:,i] = W @ spikes

	return X


def create_clus_split(nx, ny, n_centers=None):
	xv, yv = np.meshgrid(np.arange(nx), np.arange(ny))
	pts = np.stack([xv.ravel(), yv.ravel()]).T
	n = nx*ny
	if n_centers is None or n_centers == 'log':
		n_centers = 2*int(np.log2(n))
	elif n_centers == 'sqrt':
		n_centers = int(np.sqrt(n))
	ctr = np.random.choice(pts.shape[0], size=n_centers, replace=True)
	ctr = pts[ctr]
	tr_idx = np.vstack([[pt + np.array((1.25*np.random.randn(2)).astype(int)) for _ in np.arange(int(3.5*n/n_centers))] for pt in ctr])
	tr_idx = np.maximum(0, tr_idx)
	tr_idx[:,0] = cx = np.minimum(nx-1, tr_idx[:,0])
	tr_idx[:,1] = cy = np.minimum(ny-1, tr_idx[:,1])
	tr_idx = np.unique(np.ravel_multi_index(tr_idx.T, (nx,ny)))

	tr_bool = np.zeros(n).astype(bool)
	tr_bool[tr_idx] = True

	return tr_bool