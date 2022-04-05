import numpy as np

class DataGen():

  def __init__(self, 
               n=200,
               p=30,
               s=5,
               beta=None,
               eps_sigma=1,
               snr=None,
               block_corr=0.6,
               inter_corr=0.2,
               intercept=False):
    
    (self.n, self.p,
     self.eps_sigma,
     self.snr,
     self.block_corr,
     self.inter_corr) = (n, p, 
                         eps_sigma,
                         snr,
                         block_corr,
                         inter_corr)

    if p > 1:
      self.X = X = np.random.randn(n,p)
    else:
      self.X = X = np.arange(n).reshape((-1,1)) - int(n/3)
    if beta != None:
      self.beta = np.array(beta).reshape((-1,))
    else:
      if p > 1:
        beta = np.zeros(p)
        idx = np.random.choice(p,size=min(s,p),replace=False)
        # beta[idx] = np.random.uniform(-3,3,size=min(s,p))
        beta[idx] = np.random.uniform(-1,1,size=min(s,p))
        self.beta = beta
      else:
        self.beta = np.array([0.2])
    self.mu = X @ self.beta + 2*intercept

    sigvar = np.var(self.mu)
    if snr != None:
      self.eps_sigma = np.sqrt(sigvar / snr)
    else:
      self.snr = sigvar/self.eps_sigma**2

    self.reset_data()


  def reset_data(self):
    self._gen_data()
    self.unsampled_idx = np.arange(self.n).tolist()
    if hasattr(self, 'unsampled_cls_idx'):
      self.unsampled_cls_idx = np.arange(self.n_clusters).tolist()

  def _gen_data(self):

    (n, mu, 
     eps_sigma, 
     block_corr,
     inter_corr) = (self.n, self.mu, 
                    self.eps_sigma, 
                    self.block_corr,
                    self.inter_corr)
    self.block_eps = block_eps = eps_sigma*np.sqrt(block_corr)*np.random.randn(n).reshape((-1,))
    self.inter_eps = inter_eps = eps_sigma*np.sqrt(inter_corr)*np.random.randn(1)*np.ones((n,))
    self.y = y = mu + block_eps + inter_eps

    self.block_eps2 = block_eps2 = eps_sigma*np.sqrt(block_corr)*np.random.randn(n).reshape((-1,))
    self.inter_eps2 = inter_eps2 = eps_sigma*np.sqrt(inter_corr)*np.random.randn(1)*np.ones((n,))
    self.y2 = y2 = mu + block_eps2 + inter_eps2

  def get_sample(self, frac=.25, reps=100, sample_idx=None, return_replicate=False):
    n_sample = int(self.n*frac)
    if sample_idx is None:
      sample_idx = np.random.choice(self.unsampled_idx, n_sample, replace=False)
    self.unsampled_idx = [x for x in self.unsampled_idx if x not in sample_idx]

    # print("self.X.shape", self.X.shape)
    X = self.X[sample_idx,:]
    X = np.vstack([np.tile(c,(reps,1)) for c in X])

    y = self.y[sample_idx]
    self.ind_eps = ind_eps = np.random.randn(n_sample*reps)*np.sqrt(1-self.block_corr - self.inter_corr)*self.eps_sigma
    y = np.concatenate([c*np.ones(reps) for c in y]) + ind_eps
    beta = self.beta
    eps = self.block_eps[sample_idx] + self.inter_eps[sample_idx]
    eps = np.concatenate([e*np.ones(reps) for e in eps]) + ind_eps
    orig_sample_idx = sample_idx
    sample_idx = np.concatenate([np.arange(s*reps,(s+1)*reps) for s in sample_idx])

    if return_replicate:
      y2 = self.y2[orig_sample_idx]
      ind_eps2 = np.random.randn(n_sample*reps)*np.sqrt(1-self.block_corr - self.inter_corr)*self.eps_sigma
      y2 = np.concatenate([c*np.ones(reps) for c in y2]) + ind_eps2
      # eps2 = self.block_eps2[orig_sample_idx] + self.inter_eps2[orig_sample_idx]
      # eps2 = np.concatenate([e*np.ones(reps) for e in eps2]) + ind_eps

      return (sample_idx, orig_sample_idx, X, y, y2, beta, eps)

    return (sample_idx, orig_sample_idx, X, y, beta, eps)
