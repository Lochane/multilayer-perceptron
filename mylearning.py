import numpy as np
import nnfs
from nnfs.datasets import spiral_data
import math

nnfs.init() # this will set the random seed and also set float precision to 32 bits

class LayerDense:
	def __init__(self, n_inputs, n_neurons):
		self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
		self.biases = np.zeros((1, n_neurons))

	def forward(self, inputs):
		self.output = np.dot(inputs, self.weights) + self.biases

class ActivationReLU:
	def forward(self, inputs):
		self.output = np.maximum(0, inputs)

class ActivationSoftmax:
	def forward(self, inputs):
		exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
		probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
		self.output = probabilities

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

x, y = spiral_data(100, 3)

layer1 = LayerDense(2, 3)
activation1 = ActivationReLU()
layer2 = LayerDense(3, 3)  # output neurons should match class count
activation2 = ActivationSoftmax()

layer1.forward(X)
activation1.forward(layer1.output)

layer2.forward(activation1.output)
activation2.forward(layer2.output)

print(activation2.output[:5])

loss_function = LossCategoricalCrossentropy()
loss = loss_function.calculate(activation2.output, y)
print("Loss:", loss)