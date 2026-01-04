import math
import numpy as np

class StandardScaler:
	def __init__(self, features):
		self.features = features
		self.means = None
		self.stds = None

	def fit(self, data):
		if isinstance(data, np.ndarray):
			self.means = np.mean(data, axis=0)
			self.stds = np.std(data, axis=0)
		else:
			self.means = np.array([np.mean(data[feature].to_numpy()) for feature in self.features])
			self.stds = np.array([np.std(data[feature].to_numpy()) for feature in self.features])
	
	def transform(self, data):
		if isinstance(data, np.ndarray):
			# Ensure we work on a float copy to handle NaNs correctly
			x = data.astype(float, copy=True)
		else:
			x = data[self.features].to_numpy(dtype=float)

		# Replace NaNs with the corresponding feature means
		nan_mask = np.isnan(x)
		x = np.where(nan_mask, self.means, x)

		# Avoid division by zero by using 1.0 where std is zero
		std_safe = np.where(self.stds != 0, self.stds, 1.0)
		return (x - self.means) / std_safe

	def fit_transform(self, data):
		self.fit(data)
		return self.transform(data)
