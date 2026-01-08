import numpy as np
from layer import Layer
from activation import Activation

# class ActivationReLU(Activation):
# 	def __init__(self):
# 		def ReLU(inputs):
# 			self.output = np.maximum(0, inputs)
# 			return self.output
		
# 		super().__init__(ReLU, self.ReLU_prime)

class ActivationSoftmax(Layer):

	def forward(self, inputs):
		exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
		probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
		self.output = probabilities
		return self.output

	def backward(self, output_gradient, learning_rate):
		# Efficient softmax backward for batched inputs using Jacobian-vector product
		# output_gradient has same shape as self.output
		sum_grad = np.sum(output_gradient * self.output, axis=1, keepdims=True)
		return self.output * (output_gradient - sum_grad)

class ActivationSigmoid(Activation):
	def __init__(self):
		def Sigmoid(inputs):
			self.output = 1 / (1 + np.exp(-inputs))
			return self.output

		def Sigmoid_prime(inputs):
			s = Sigmoid(inputs)
			return s * (1 - s)
		
		super().__init__(Sigmoid, Sigmoid_prime)