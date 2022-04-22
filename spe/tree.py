from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.validation import check_is_fitted

class LinearSelector(ABC):
	@abstractmethod
	def get_linear_smoother(self, X):
		pass

	@abstractmethod
	def get_selected_X(self, X):
		pass

class Tree(LinearSelector, DecisionTreeRegressor):
	def get_linear_smoother(self, X):
		Z = self.get_selected_X(X)
		return Z @ np.linalg.inv(Z.T @ Z) @ Z.T

	def get_selected_X(self, X):
		check_is_fitted(self)

		leaf_nodes = self.apply(X)
		_, indices = np.unique(leaf_nodes, return_inverse=True)

		n_leaves = self.get_n_leaves()
		n = X.shape[0]

		Z = np.zeros((n, n_leaves))
		Z[np.arange(n),indices] = 1

		return Z










