from abc import ABCMeta, abstractmethod

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils.validation import check_is_fitted

from joblib import Parallel


## code from sklearn, only need to change a small part of fit function

class BlurredForest(RandomForestRegressor):
	def get_linear_smoother(self, X, X_pred=None):
		Gs = self.get_group_X(X)
		if X_pred is None:
			G_preds = Gs
		else:
			G_preds = self.get_group_X(X_pred)
		
		return [G_pred @ np.linalg.inv(G.T @ G) @ G.T for (G,G_pred) in zip(Gs, G_preds)]

	def get_group_X(self, X):
		check_is_fitted(self)

		leaf_node_array = self.apply(X)

		stored_n_leaves = hasattr(self, 'n_leaves')

		# if not stored_n_leaves:
		# 	self.n_leaves = np.zeros(leaf_node_array.shape[1]).astype(int)

		Gs = []
		for i in np.arange(leaf_node_array.shape[1]):
			leaf_nodes = leaf_node_array[:,i]
			_, indices = np.unique(leaf_nodes, return_inverse=True)

			# if not stored_n_leaves:
			# 	self.n_leaves[i] = n_leaves = np.amax(indices) + 1
			# else:
			# 	n_leaves = self.n_leaves[i]
			n_leaves = np.amax(indices) + 1
			n = X.shape[0]

			G = np.zeros((n, n_leaves))
			G[np.arange(n),indices] = 1
			Gs.append(G)

		return Gs
