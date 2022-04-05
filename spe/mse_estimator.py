import numpy as np
from numpy.core.numeric import allclose
import pandas as pd
from scipy.linalg import toeplitz, block_diag
from scipy.spatial.distance import cdist
from itertools import product
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, LinearRegression, Lasso
from sklearn.model_selection import cross_validate, GroupKFold, KFold
from sklearn.cluster import KMeans

from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

import plotly.express as px

from .data_generator import DataGen
from .relaxed_lasso import RelaxedLasso

class MSESimulator(object):

  def _get_kfcv_mse(self,
                    model,
                    X,
                    y,
                    k):
    kfcv_res = cross_validate(model,
                              X,
                              y, 
                              scoring='neg_mean_squared_error', 
                              cv=KFold(k, shuffle=True), 
                              error_score='raise')
    return -np.mean(kfcv_res['test_score'])


  def _get_spcv_mse(self,
                    model,
                    X,
                    y,
                    k):
    groups = KMeans(n_clusters=k).fit(X).labels_
    spcv_res = cross_validate(model,
                              X,
                              y, 
                              scoring='neg_mean_squared_error', 
                              cv=GroupKFold(k), 
                              groups=groups)

    return -np.mean(spcv_res['test_score'])

  def _get_true_mse(self,
                    model,
                    X,
                    X_pred,
                    y,
                    y_pred):
    model.fit(X, y)

    pred = model.predict(X_pred)
    resids = (y_pred - pred)**2
    return np.mean(resids)

  def _get_gmcp_mse(self,
                    model, 
                    X_tr,
                    X_ts, 
                    y_tr,
                    y_ts, 
                    idx_tr,
                    idx_ts, 
                    model_type, 
                    pred_type, 
                    Sigma_s, 
                    fit_intercept):

    model.fit(X_tr, y_tr)

    if model_type == 'lasso':
      XE_tr = X_tr[:, model.E_]
      XE_ts = X_ts[:, model.E_]
    elif model_type == 'linear':
      XE_tr = X_tr
      XE_ts = X_ts

    if fit_intercept:
      X_mat = np.hstack([np.ones((XE_tr.shape[0],1)),XE_tr])
      X_mat_ts = np.hstack([np.ones((XE_ts.shape[0],1)),XE_ts])
    else:
      X_mat = XE_tr
      X_mat_ts = XE_ts
    XtXinv = np.linalg.inv(X_mat.T.dot(X_mat)).dot(X_mat.T)

    H = X_mat.dot(XtXinv)
    H_ts = X_mat_ts.dot(XtXinv)
    idxs = np.concatenate([idx_tr, idx_ts])
    if pred_type == 'test':
      gmcp_pr = model.predict(X_ts)
      gmcp_resids = np.mean((y_ts - gmcp_pr)**2)
      # H_e = np.hstack([H_ts,np.zeros((H_ts.shape[0],X_ts.shape[0]))])
      # Sigma_ss = Sigma_s[idxs,:][:, idx_ts]
      H_e = H_ts
      Sigma_ss = Sigma_s[idx_tr,:][:, idx_ts]
    elif pred_type == 'train':
      gmcp_pr = model.predict(X_tr)
      gmcp_resids = np.mean((y_tr - gmcp_pr)**2)
      H_e = H
      Sigma_ss = Sigma_s[idx_tr,:][:, idx_tr]

    return gmcp_resids + 2*np.diag(H_e @ Sigma_ss).sum() / len(y_ts)
            #gmcp_resids + 2*np.diag(H_e2 @ Sigma_ss2).sum() / len(y_ts))

  def _get_naive_rand_mse(self,
                          model,
                          X_tr,
                          X_pr, 
                          y_sel,
                          y_fit,
                          y_pr):
    
    model.fit(X_tr,
              y_sel,
              y_fit)

    preds = model.predict(X_pr)
    return np.mean((y_pr - preds)**2)

  def _get_full_correction(self,
                           X_tr,
                           X_ts, 
                           Sigma, 
                           A_perp, 
                           n_ts,
                           fit_intercept):

    if fit_intercept:
      X_mat = np.hstack([np.ones((X_tr.shape[0],1)),X_tr])
      X_mat_ts = np.hstack([np.ones((X_ts.shape[0],1)),X_ts])
    else:
      X_mat = X_tr
      X_mat_ts = X_ts
    XtXinvXt = np.linalg.inv(X_mat.T.dot(X_mat)).dot(X_mat.T)

    H_ts = X_mat_ts.dot(XtXinvXt)
    # idxs = np.concatenate([idx_tr, idx_ts])
    # H_e = np.hstack([H_ts,np.zeros((H_ts.shape[0],X_ts.shape[0]))])
    # Sigma_ss = Sigma_s[idxs,:][:, idx_ts]
    
    return 2*np.diag(H_ts @ A_perp @ Sigma).sum() / n_ts

  def _get_rand_mse(self,
                    model,
                    nboot,
                    X,
                    y,
                    y2,
                    idx_tr,
                    idx_ts, 
                    Sigma_t,
                    Chol_o,
                    Prec_o,
                    ret_mse,
                    fit_intercept):
    
    X_tr = X[idx_tr,:]
    X_ts = X[idx_ts,:]
    y_tr = y[idx_tr]
    y_ts = y[idx_ts]

    full_mse = np.zeros(nboot)
    nonhonest_mse = np.zeros(nboot)
    honest_mse = np.zeros(nboot)


    Cov_ot = Sigma_t @ Prec_o
    A_perp_inv = np.eye(Sigma_t.shape[0]) + Cov_ot
    A_perp = np.linalg.inv(A_perp_inv)
    Sigma_wp = A_perp_inv @ Sigma_t
    wp_correction = (Sigma_t[idx_ts,idx_ts].sum() * ret_mse
                      - Sigma_wp[idx_ts,idx_ts].sum()) / len(y_ts)

    for b in np.arange(nboot):
      omega = Chol_o @ np.random.randn(len(y))
      w = y + omega
      w_perp = y_ts - (Cov_ot @ omega)[idx_ts]
      omega_2 = Chol_o @ np.random.randn(len(y))
      w0 = w + omega_2
      w1 = w - (Cov_ot @ omega_2)

      # model.fit(X_tr, w, y_tr) # full
      # model.fit(X_tr, w, w) # nonhonest
      # model.fit(X_tr, w0, w1) # honest

      full_naive_mse = self._get_naive_rand_mse(model,
                                                X,
                                                X_ts,
                                                w,
                                                y,
                                                w_perp)
      # A_perp_inv = np.eye(Sigma_t.shape[0]) + Cov_ot
      # A_perp = np.linalg.inv(A_perp_inv)
      # Sigma_wp = A_perp_inv @ Sigma_t

      full_correction = self._get_full_correction(X[:,model.E_], 
                                                  X_ts[:,model.E_], 
                                                  Sigma_wp[:,idx_ts],
                                                  A_perp,
                                                  len(y_ts),
                                                  fit_intercept)
      
      nonhonest_naive_mse = self._get_naive_rand_mse(model,
                                                     X,
                                                     X_ts,
                                                     w,
                                                     w,
                                                     w_perp)

      honest_naive_mse = self._get_naive_rand_mse(model,
                                                  X,
                                                  X_ts,
                                                  w0,
                                                  w1,
                                                  w_perp)


      # Sigma_wp = (np.eye(Sigma_t.shape[0]) + Cov_ot)[idx_ts,:] @ Sigma_t[:,idx_ts]
      # wp_correction = (np.diag(Sigma_t[idx_ts,:][:,idx_ts]).sum() 
      #                   - np.diag(Sigma_wp).sum()) / len(y_ts)
      # wp_correction = (np.diag(Sigma_t[idx_ts,:][:,idx_ts]).sum() 
      #                   - np.diag(Sigma_wp[idx_ts,:][:,idx_ts]).sum()) / len(y_ts)

      # wp_correction = (Sigma_t[idx_ts,idx_ts].sum() * ret_mse
      #                   - Sigma_wp[idx_ts,idx_ts].sum()) / len(y_ts)
      
      full_mse[b] = full_naive_mse# + wp_correction + full_correction
      nonhonest_mse[b] = nonhonest_naive_mse# + wp_correction
      honest_mse[b] = honest_naive_mse# + wp_correction

    
    true_mse = self._get_naive_rand_mse(model,
                                        X,
                                        X_ts,
                                        w,
                                        y,
                                        y2[idx_ts])

    return (full_mse.mean() + wp_correction + full_correction,
            nonhonest_mse.mean() + wp_correction,
            honest_mse.mean() + wp_correction,
            true_mse)

  def cv_compare(self,
                 niter=100,
                 n=200,
                 p=50,
                 s=5,
                 reps=100,
                 alpha=1.,
                 nboot=100,
                 eps_sigma=1,
                 snr=None,
                 block_corr=0.6,
                 inter_corr=0.,
                 train_frac=0.1,
                 test_frac=0.1,
                 k=5,
                 fit_intercept=False,
                 model_type='linear',
                 lambd=1.,
                 pred_type='train',
                 ret_mse=True):

    (self.n,
     self.p,
     self.s,
     self.snr,
     self.reps,
     self.niter,
     self.nboot,
     self.eps_sigma,
     self.block_corr,
     self.inter_corr,
     self.ret_mse) = (n,
                      p,
                      s,
                      snr,
                      reps,
                      niter,
                      nboot,
                      eps_sigma,
                      block_corr,
                      inter_corr,
                      ret_mse)

    n_train = int(n*train_frac*reps)
    n_tot = int(n*(train_frac + test_frac)*reps)

    Sigma_s = np.ones((n*reps, n*reps))*inter_corr
    Sigma_s += block_diag(*[np.ones((reps,reps))*block_corr for _ in np.arange(n)])
    Sigma_s += np.eye(n*reps,n*reps)*(1 - block_corr - inter_corr)
    Sigma_s *= eps_sigma

    self.true_mse = np.zeros(niter)
    self.kfcv_mse = np.zeros(niter)
    self.spcv_mse = np.zeros(niter)
    self.gmcp_mse = np.zeros(niter)
    self.frft_mse = np.zeros(niter)
    self.nhnst_mse = np.zeros(niter)
    self.hnst_mse = np.zeros(niter)

    self.data_gen = DataGen(n,
                            p,
                            s,
                            snr=snr,
                            eps_sigma=eps_sigma, 
                            block_corr=block_corr, 
                            inter_corr=inter_corr, 
                            intercept=fit_intercept)
    
    self.eps_sigma = eps_sigma = self.data_gen.eps_sigma
    self.snr = snr = self.data_gen.snr
    
    if model_type == 'linear':
      model = LinearRegression(fit_intercept=fit_intercept)
    elif model_type == 'lasso':
      model = RelaxedLasso(lambd=lambd, 
                           fit_intercept=fit_intercept)

    n_rand = n_tot if pred_type == 'test' else n_train
    Sigma_o = np.eye(n_rand)*alpha*eps_sigma**2
    Prec_o = np.eye(n_rand)/(alpha*eps_sigma**2)
    
    Chol_o = np.linalg.cholesky(Sigma_o)

    gen_idx_tr = None
    gen_idx_ts = None

    for i in np.arange(niter):
      if i % 25 == 0: print(i)
      if i > 0:
        self.data_gen.reset_data()
      
      if train_frac > 0:
        (idx_tr,
         orig_idx_tr,
         X_tr, 
         y_tr,
         y_tr2,
         beta,
         eps_tr) = self.data_gen.get_sample(frac=train_frac, 
                                            reps=reps, 
                                            sample_idx=gen_idx_tr,
                                            return_replicate=True)
        if gen_idx_tr is None:
          gen_idx_tr = orig_idx_tr
      
      if test_frac > 0:
        (idx_ts,
         orig_idx_ts,
         X_ts, 
         y_ts,
         y_ts2,
         _,
         eps_ts) = self.data_gen.get_sample(frac=test_frac, 
                                            reps=reps, 
                                            sample_idx=gen_idx_ts,
                                            return_replicate=True)
        if gen_idx_ts is None:
          gen_idx_ts = orig_idx_ts
      
      self.kfcv_mse[i] = self._get_kfcv_mse(model, X_tr, y_tr, k)

      self.spcv_mse[i] = self._get_spcv_mse(model, X_tr, y_tr, k)
      
      if model_type == 'linear':
        self.true_mse[i] = (self._get_true_mse(model,
                                              X_tr,
                                              X_ts,
                                              y_tr,
                                              y_ts2) if pred_type == 'test' else
                            self._get_true_mse(model,
                                               X_tr,
                                               X_tr,
                                               y_tr,
                                               y_tr2)

        self.gmcp_mse[i] = self._get_gmcp_mse(model,
                                              X_tr,
                                              X_ts, 
                                              y_tr,
                                              y_ts, 
                                              idx_tr,
                                              idx_ts, 
                                              model_type, 
                                              pred_type,
                                              Sigma_s,
                                              fit_intercept)
      
      if model_type == 'lasso':
        if pred_type == 'test':
          idxs = np.concatenate([idx_tr, idx_ts])
          (self.frft_mse[i], 
          self.nhnst_mse[i], 
          self.hnst_mse[i],
          self.true_mse[i]) = self._get_rand_mse(model, nboot,
                                                 np.vstack([X_tr, X_ts]), 
                                                 np.concatenate([y_tr,y_ts]), 
                                                 np.concatenate([y_tr2,y_ts2]), 
                                                 np.arange(len(y_tr)), 
                                                 np.arange(len(y_tr),len(y_tr) + len(y_ts)),
                                                 Sigma_s[idxs,:][:,idxs],
                                                 Chol_o,
                                                 Prec_o,
                                                 ret_mse,
                                                 fit_intercept)
        else:
          (self.frft_mse[i], 
           self.nhnst_mse[i], 
           self.hnst_mse[i],
           self.true_mse[i]) = self._get_rand_mse(model, nboot,
                                                  X_tr, 
                                                  y_tr, 
                                                  y_tr2, 
                                                  np.arange(len(y_tr)), 
                                                  np.arange(len(y_tr)),
                                                  Sigma_s[idx_tr,:][:,idx_tr],
                                                  Chol_o,
                                                  Prec_o,
                                                  ret_mse,
                                                  fit_intercept)
        
    return (self.true_mse, 
            self.kfcv_mse, 
            self.spcv_mse, 
            self.gmcp_mse,
            self.frft_mse,
            self.nhnst_mse,
            self.hnst_mse)

  def gen_boxplot(self):
    
    mse_df = pd.DataFrame({'kfcv_mse': (self.kfcv_mse)/self.true_mse,
                          'spcv_mse': (self.spcv_mse)/self.true_mse,
                          'gmcp_mse': (self.gmcp_mse)/self.true_mse,
                          'frft_mse': (self.frft_mse)/self.true_mse,
                          'nhnst_mse': (self.nhnst_mse)/self.true_mse,
                          'hnst_mse': (self.hnst_mse)/self.true_mse})
    mse_df['idx'] = mse_df.index.values
    mse_df.set_index('idx')
    mse_df.reset_index()
    long_df = pd.melt(mse_df, id_vars='idx', value_vars=['spcv_mse',
                                                         'kfcv_mse',
                                                         'gmcp_mse', 
                                                         'frft_mse',
                                                         'nhnst_mse',
                                                         'hnst_mse'])
    long_df.drop(columns='idx',inplace=True)

    fig = px.box(long_df, 
                  x='variable',
                  y='value',
                  color='variable',
                  points='all',
                  title=f'{self.n*self.reps}x{self.p}, {self.reps} repls, {self.niter} its, {self.block_corr} blk corr, {self.inter_corr} intr corr',
                  labels={'variable': 'Validation Method',
                          'value': 'Relative MSE'})
    fig.update_traces(boxmean=True)
    fig.add_hline(y=1.)
    return fig
  


############################################################################################################################################
############################################################################################################################################
############################################################################################################################################



# import numpy as np
# from numpy.core.numeric import allclose
# import pandas as pd
# from scipy.linalg import toeplitz, block_diag
# from scipy.spatial.distance import cdist
# from itertools import product
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.linear_model import Ridge, LinearRegression, Lasso
# from sklearn.model_selection import cross_validate, GroupKFold, KFold
# from sklearn.cluster import KMeans

# from sklearn.base import BaseEstimator
# from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

# import plotly.express as px

# from .data_generator import DataGen
# from .relaxed_lasso import RelaxedLasso

# class MSESimulator(object):

#   def _get_kfcv_mse(self, model, X, y, k):
#     kfcv_res = cross_validate(model, X, y, 
#                               scoring='neg_mean_squared_error', 
#                               cv=KFold(k, shuffle=True), 
#                               error_score='raise')
#     return -np.mean(kfcv_res['test_score'])


#   def _get_spcv_mse(self, model, X, y, k):
#     groups = KMeans(n_clusters=k).fit(X).labels_
#     spcv_res = cross_validate(model, X, y, 
#                               scoring='neg_mean_squared_error', 
#                               cv=GroupKFold(k), 
#                               groups=groups)

#     return -np.mean(spcv_res['test_score'])

#   def _get_true_mse(self, model, X, X_pred, y, y_pred):
#     model.fit(X, y)

#     pred = model.predict(X_pred)
#     resids = (y_pred - pred)**2
#     return np.mean(resids)

#   def _get_gmcp_mse(self, model, 
#                   X_tr, X_ts, 
#                   y_tr, y_ts, 
#                   idx_tr, idx_ts, 
#                   model_type, 
#                   pred_type, 
#                   Sigma_s, 
#                   fit_intercept):
#     model.fit(X_tr, y_tr)

#     if model_type == 'lasso':
#       XE_tr = X_tr[:, model.E_]
#       XE_ts = X_ts[:, model.E_]
#     elif model_type == 'linear':
#       XE_tr = X_tr
#       XE_ts = X_ts

#     if fit_intercept:
#       X_mat = np.hstack([np.ones((XE_tr.shape[0],1)),XE_tr])
#       X_mat_ts = np.hstack([np.ones((XE_ts.shape[0],1)),XE_ts])
#     else:
#       X_mat = XE_tr
#       X_mat_ts = XE_ts
#     XtXinv = np.linalg.inv(X_mat.T.dot(X_mat)).dot(X_mat.T)

#     H = X_mat.dot(XtXinv)
#     H_ts = X_mat_ts.dot(XtXinv)
#     idxs = np.concatenate([idx_tr, idx_ts])
#     if pred_type == 'test':
#       gmcp_pr = model.predict(X_ts)
#       gmcp_resids = np.mean((y_ts - gmcp_pr)**2)
#       # H_e = np.hstack([H_ts,np.zeros((H_ts.shape[0],X_ts.shape[0]))])
#       # Sigma_ss = Sigma_s[idxs,:][:, idx_ts]
#       H_e = H_ts
#       Sigma_ss = Sigma_s[idx_tr,:][:, idx_ts]
#     elif pred_type == 'train':
#       gmcp_pr = model.predict(X_tr)
#       gmcp_resids = np.mean((y_tr - gmcp_pr)**2)
#       H_e = H
#       Sigma_ss = Sigma_s[idx_tr,:][:, idx_tr]

#     return gmcp_resids + 2*np.diag(H_e @ Sigma_ss).sum() / len(y_ts)
#             #gmcp_resids + 2*np.diag(H_e2 @ Sigma_ss2).sum() / len(y_ts))

#   def _get_naive_rand_mse(self, model, X_tr, X_pr, 
#                           y_sel, y_fit, y_pr):
    
#     model.fit(X_tr, y_sel, y_fit)

#     preds = model.predict(X_pr)
#     return np.mean((y_pr - preds)**2)

#   def _get_full_correction(self, X_tr, X_ts, 
#                           Sigma, 
#                           A_perp, 
#                           n_ts,
#                           fit_intercept):

#     if fit_intercept:
#       X_mat = np.hstack([np.ones((X_tr.shape[0],1)),X_tr])
#       X_mat_ts = np.hstack([np.ones((X_ts.shape[0],1)),X_ts])
#     else:
#       X_mat = X_tr
#       X_mat_ts = X_ts
#     XtXinvXt = np.linalg.inv(X_mat.T.dot(X_mat)).dot(X_mat.T)

#     H_ts = X_mat_ts.dot(XtXinvXt)
#     # idxs = np.concatenate([idx_tr, idx_ts])
#     # H_e = np.hstack([H_ts,np.zeros((H_ts.shape[0],X_ts.shape[0]))])
#     # Sigma_ss = Sigma_s[idxs,:][:, idx_ts]
    
#     return 2*np.diag(H_ts @ A_perp @ Sigma).sum() / n_ts

#   def _get_rand_mse(self, model, X, y, y2,
#                   idx_tr, idx_ts, 
#                   Sigma_t, Chol_o,
#                   Prec_o,# Sigma_wp,
#                   #  A_perp,
#                   fit_intercept):
    
#     X_tr = X[idx_tr,:]
#     X_ts = X[idx_ts,:]
#     y_tr = y[idx_tr]
#     y_ts = y[idx_ts]
#     omega = Chol_o @ np.random.randn(len(y))
#     # omega_tr = omega[idx_tr]
#     # omega_ts = omega[idx_ts]
#     w = y + omega
#     # w_tr = w[idx_tr]
#     # w_ts = w[idx_ts]
#     Cov_ot = Sigma_t @ Prec_o
#     w_perp = y_ts - (Cov_ot @ omega)[idx_ts]
#     omega_2 = Chol_o @ np.random.randn(len(y))
#     w0 = w + omega_2
#     w1 = w - (Cov_ot @ omega_2)#[idx_ts,:]

#     # model.fit(X_tr, w, y_tr) # full
#     # model.fit(X_tr, w, w) # nonhonest
#     # model.fit(X_tr, w0, w1) # honest
#     full_naive_mse = self._get_naive_rand_mse(model, X, X_ts, w, y, w_perp)
#     A_perp = np.linalg.inv(np.eye(Sigma_t.shape[0]) + Cov_ot)
#     Sigma_wp = (np.eye(Sigma_t.shape[0]) + Cov_ot) @ Sigma_t
#     full_correction = self._get_full_correction(X[:,model.E_], 
#                                           X_ts[:,model.E_], 
#                                           Sigma_wp[:,idx_ts],
#                                           A_perp,#[:,idx_ts],
#                                           len(y_ts),
#                                           fit_intercept)
    
#     nonhonest_naive_mse = self._get_naive_rand_mse(model, X, X_ts, w, w, w_perp)
#     honest_naive_mse = self._get_naive_rand_mse(model, X, X_ts, w0, w1, w_perp)


#     # Sigma_wp = (np.eye(Sigma_t.shape[0]) + Cov_ot)[idx_ts,:] @ Sigma_t[:,idx_ts]
#     # wp_correction = (np.diag(Sigma_t[idx_ts,:][:,idx_ts]).sum() 
#     #                   - np.diag(Sigma_wp).sum()) / len(y_ts)
#     # wp_correction = (np.diag(Sigma_t[idx_ts,:][:,idx_ts]).sum() 
#     #                   - np.diag(Sigma_wp[idx_ts,:][:,idx_ts]).sum()) / len(y_ts)
#     wp_correction = (Sigma_t[idx_ts,idx_ts].sum() 
#                       - Sigma_wp[idx_ts,idx_ts].sum()) / len(y_ts)
    
#     full_mse = full_naive_mse + wp_correction + full_correction
#     nonhonest_mse = nonhonest_naive_mse + wp_correction
#     honest_mse = honest_naive_mse + wp_correction
#     true_mse = self._get_naive_rand_mse(model, X, X_ts, w, y, y2[idx_ts])

#     return full_mse, nonhonest_mse, honest_mse, true_mse

#     # if model_type == 'lasso':
#     #   XE_tr = X_tr[:, model.E_]
#     #   XE_ts = X_ts[:, model.E_]
#     # elif model_type == 'linear':
#     #   XE_tr = X_tr
#     #   XE_ts = X_ts

#     # if fit_intercept:
#     #   X_mat = np.hstack([np.ones((XE_tr.shape[0],1)),XE_tr])
#     #   X_mat_ts = np.hstack([np.ones((XE_ts.shape[0],1)),XE_ts])
#     # else:
#     #   X_mat = XE_tr
#     #   X_mat_ts = XE_ts
#     # XtXinv = np.linalg.inv(X_mat.T.dot(X_mat)).dot(X_mat.T)

#     # H = X_mat.dot(XtXinv)
#     # H_ts = X_mat_ts.dot(XtXinv)
#     # idxs = np.concatenate([idx_tr, idx_ts])
#     # if pred_type == 'test':
#     #   gmcp_pr = model.predict(X_ts)
#     #   gmcp_resids = np.mean((w_perp - gmcp_pr)**2)
#     #   H_e = np.hstack([H_ts,np.zeros((H_ts.shape[0],X_ts.shape[0]))])
#     #   Sigma_ss = Sigma_s[idxs,:][:, idx_ts]
#     # elif pred_type == 'train':
#     #   gmcp_pr = model.predict(X_tr)
#     #   gmcp_resids = np.mean((y_tr - gmcp_pr)**2)
#     #   H_e = H
#     #   Sigma_ss = Sigma_s[idx_tr,:][:, idx_tr]
    
#     # return gmcp_resids + 2*np.diag(H_e @ Sigma_ss).sum() / len(y_ts)

#   def cv_compare(self, niter=100, n=200,
#                 p=50,
#                 reps=100,
#                 alpha=1.,
#                 eps_sigma=1,
#                 block_corr=0.6,
#                 inter_corr=0.,
#                 train_frac=0.1,
#                 test_frac=0.1,
#                 k=5,
#                 fit_intercept=True,
#                 model_type='linear',
#                 lambd=1.,
#                 pred_type='train'):

#     (self.n,
#      self.p,
#      self.reps,
#      self.niter,
#      self.eps_sigma,
#      self.block_corr,
#      self.inter_corr) = (n,
#                          p,
#                          reps,
#                          niter,
#                          eps_sigma,
#                          block_corr,
#                          inter_corr)
#     n_train = int(n*train_frac*reps)
#     n_tot = int(n*(train_frac + test_frac)*reps)

#     Sigma_s = np.ones((n*reps, n*reps))*inter_corr
#     Sigma_s += block_diag(*[np.ones((reps,reps))*block_corr for _ in np.arange(n)])
#     Sigma_s += np.eye(n*reps,n*reps)*(1 - block_corr - inter_corr)
#     Sigma_s *= eps_sigma

#     self.true_mse = np.zeros(niter)
#     self.kfcv_mse = np.zeros(niter)
#     self.spcv_mse = np.zeros(niter)
#     self.gmcp_mse = np.zeros(niter)
#     self.frft_mse = np.zeros(niter)
#     self.nhnst_mse = np.zeros(niter)
#     self.hnst_mse = np.zeros(niter)

#     data_gen = DataGen(n, p, 
#                       eps_sigma=eps_sigma, 
#                       block_corr=block_corr, 
#                       inter_corr=inter_corr, 
#                       intercept=fit_intercept)
    
#     if model_type == 'linear':
#       model = LinearRegression(fit_intercept=fit_intercept)
#     elif model_type == 'lasso':
#       model = RelaxedLasso(lambd=lambd, 
#                            fit_intercept=fit_intercept)

#     Sigma_o = np.eye(n_tot)*alpha*eps_sigma**2
#     Chol_o = np.linalg.cholesky(Sigma_o)
#     Prec_o = np.eye(n_tot)/(alpha*eps_sigma**2)

#     Sigma_wp = Sigma_s + Sigma_s.dot(Sigma_s) / (alpha*eps_sigma**2)
#     gen_idx_tr = None
#     gen_idx_ts = None

#     for i in np.arange(niter):
#       if i % 25 == 0: print(i)
#       if i > 0:
#         data_gen.reset_data()
      
#       (idx_tr, orig_idx_tr, X_tr, 
#       y_tr, y_tr2, beta, eps_tr) = data_gen.get_sample(frac=train_frac, 
#                                                         reps=reps, 
#                                                         sample_idx=gen_idx_tr,
#                                                         return_replicate=True)
#       if gen_idx_tr is None:
#         gen_idx_tr = orig_idx_tr
      
#       # tr_bool = np.zeros(n*reps, dtype=bool)
#       # tr_bool[idx_tr] = True
      
#       (idx_ts, orig_idx_ts, X_ts, 
#       y_ts, y_ts2, _, eps_ts) = data_gen.get_sample(frac=test_frac, 
#                                                     reps=reps, 
#                                                     sample_idx=gen_idx_ts,
#                                                     return_replicate=True)
#       if gen_idx_ts is None:
#         gen_idx_ts = orig_idx_ts
      
#       self.kfcv_mse[i] = self._get_kfcv_mse(model, X_tr, y_tr, k)

#       self.spcv_mse[i] = self._get_spcv_mse(model, X_tr, y_tr, k)
      
#       if model_type == 'linear':
#         self.true_mse[i] = self._get_true_mse(model, X_tr, X_ts, y_tr, y_ts2) if \
#                         pred_type == 'test' else self._get_true_mse(model, X_tr, X_tr, y_tr, y_tr2)

#         self.gmcp_mse[i] = self._get_gmcp_mse(model, X_tr, X_ts, 
#                                   y_tr, y_ts, 
#                                   idx_tr, idx_ts, 
#                                   model_type, 
#                                   pred_type, Sigma_s,
#                                   fit_intercept)
      
#       if model_type == 'lasso':
#         idxs = np.concatenate([idx_tr, idx_ts])
#         (self.frft_mse[i], 
#         self.nhnst_mse[i], 
#         self.hnst_mse[i],
#         self.true_mse[i]) = self._get_rand_mse(model, np.vstack([X_tr, X_ts]), 
#                                     np.concatenate([y_tr,y_ts]), 
#                                     np.concatenate([y_tr2,y_ts2]), 
#                                     np.arange(len(y_tr)), 
#                                     np.arange(len(y_tr),len(y_tr) + len(y_ts)),
#                                     Sigma_s[idxs,:][:,idxs],
#                                     Chol_o,
#                                     Prec_o,
#                                     # Sigma_wp[idxs,:][:,idxs],
#                                     fit_intercept)
        

#     return self.true_mse, self.kfcv_mse, self.spcv_mse, self.gmcp_mse, self.frft_mse, self.nhnst_mse, self.hnst_mse

#   def gen_boxplot(self):
    
#     mse_df = pd.DataFrame({'kfcv_mse': (self.kfcv_mse)/self.true_mse,
#                           'spcv_mse': (self.spcv_mse)/self.true_mse,
#                           'gmcp_mse': (self.gmcp_mse)/self.true_mse,
#                           'frft_mse': (self.frft_mse)/self.true_mse,
#                           'nhnst_mse': (self.nhnst_mse)/self.true_mse,
#                           'hnst_mse': (self.hnst_mse)/self.true_mse})
#     mse_df['idx'] = mse_df.index.values
#     mse_df.set_index('idx')
#     mse_df.reset_index()
#     long_df = pd.melt(mse_df, id_vars='idx', value_vars=['spcv_mse','kfcv_mse', 'gmcp_mse', 
#                                                         'frft_mse', 'nhnst_mse', 'hnst_mse'])
#     long_df.drop(columns='idx',inplace=True)

#     fig = px.box(long_df, 
#                   x='variable',
#                   y='value',
#                   color='variable',
#                   points='all',
#                   title=f'{self.n*self.reps}x{self.p}, {self.reps} repls, {self.niter} its, {self.block_corr} blk corr, {self.inter_corr} intr corr',
#                   labels={'variable': 'Validation Method',
#                           'value': 'Relative MSE'})
#     fig.update_traces(boxmean=True)
#     fig.add_hline(y=1.)
#     return fig
  







