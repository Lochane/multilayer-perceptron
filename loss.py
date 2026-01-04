import numpy as np

class Loss:
	def calculate(self, outputs, y):
		sample_losses = self.forward(outputs, y)
		data_loss = np.mean(sample_losses)
		return data_loss

class LossCategoricalCrossentropy(Loss):
	def forward(self, y_pred, y_true):
		samples = len(y_pred)
		y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

		if len(y_true.shape) == 1:
			correct_confidances = y_pred_clipped[range(samples), y_true]
		elif len(y_true.shape) == 2:
			correct_confidances = np.sum(y_pred_clipped * y_true, axis=1)

		negative_log_likelihoods = -np.log(correct_confidances)
		return negative_log_likelihoods