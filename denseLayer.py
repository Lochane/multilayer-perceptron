import numpy as np
from layer import Layer

class DenseLayer(Layer):
	def __init__(self, input_size, output_size, weights_initializer="random"):
		super().__init__()
		self.input_size = input_size
		self.output_size = output_size
		self.weights_initializer = weights_initializer

		self.weights = None
		self.bias = None

		if self.weights_initializer == "heUniform":
			limit = np.sqrt(6 / input_size)
			self.weights = 0.10 * np.random.uniform(-limit, limit, (input_size, output_size))
		else:
			self.weights = 0.10 * np.random.randn(input_size, output_size)
		self.bias = np.zeros((1, output_size))

	def forward(self, inputs):
		# Ensure numerical inputs even if upstream provided object/None values
		self.input = np.asarray(inputs, dtype=float)
		self.output = np.dot(self.input, self.weights) + self.bias
		return self.output

	def backward(self, output_gradient, learning_rate):
		pass