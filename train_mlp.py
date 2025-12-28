import pandas as pd
import numpy as np
from utils import stats_tools as st

INCLUDE_FEATURES = ["texture_mean", "perimeter_mean", "compactness_mean", "concavity_mean","concave_points_mean"]

class DenseLayer(self, input_size, output_size, activation):
	self.weights = np.random.randn(input_size, output_size)
	self.bias = np.zeros((1, output_size))
	self.activation = activation

	def activate(self, x):
		if self.activation == "sigmoid":
			return 1 / (1 + np.exp(-x))
		elif self.activation == "softmax":
			exp_x = np.exp(x - np.max(x))
			return exp_x / exp_x.sum(axis=1, keepdims=True)
		else:
			raise ValueError("Unsupported activation function")

# class MLP:
# 	# Initialize weights and biases: Randomly initialize weights and set biases to zero.
# 	def __init__(self, input_size, hidden_size, output_size):
# 		self.weights_input_hidden = np.random.randn(input_size, hidden_size)
# 		self.weights_hidden_output = np.random.randn(hidden_size, output_size)
# 		self.bias_hidden = np.zeros((1, hidden_size))
# 		self.bias_output = np.zeros((1, output_size))

# 	def sigmoide(score):
#     	return 1 / (1 + np.exp(-score))

# 	def softmax(self, x):
# 		exp_x = np.exp(x - np.max(x))
# 		return exp_x / exp_x.sum(axis=1, keepdims=True)

# 	# Compute hidden and output layer inputs and outputs: Apply activation functions to compute the activations.
# 	def forward(self, x):
# 		self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
# 		self.hidden_output = self.sigmoide(self.hidden_input)
# 		self.final_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
# 		self.final_output = self.softmax(self.final_input)
# 		return self.final_output

# 	# Compute errors and update weights and biases: Adjust the weights and biases using the gradient descent algorithm.
# 	def backward(self, x, y, output, learning_rate):
# 		output_error = output - y
# 		hidden_error = np.dot(output_error, self.weights_hidden_output.T) * self.hidden_output * (1 - self.hidden_output)

# 		self.weights_hidden_output -= learning_rate * np.dot(self.hidden_output.T, output_error)
# 		self.bias_output -= learning_rate * np.sum(output_error, axis=0, keepdims=True)
# 		self.weights_input_hidden -= learning_rate * np.dot(x.T, hidden_error)
# 		self.bias_hidden -= learning_rate * np.sum(hidden_error, axis=0, keepdims=True)
	
# 	def tain(self, X, y, epochs, learning_rate):
# 		for epoch in range(epochs):
# 			output = self.forward(X)
# 			self.backward(X, y, output, learning_rate)
# 			if (epoch + 1) % 100 == 0:
# 				loss_value = -np.sum(y * np.log(output)) / x.shape[0]
# 				print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss_value}")

def neurons(x, weights, bias):
	return sigmoide(sum(x[i] * weights[i] for i in range(len(x))) + bias)

def loss(y_true, y_pred):
	n= len(y_true)
	return -sum(y_true[i] * np.log(y_pred[i]) + (1 - y_true[i]) * np.log(1 - y_pred[i]) for i in range(n)) / n

def main(argv=None):
	#! check if data file exists
	train_df = pd.read_csv("data_splits/train_data.csv") 


	# Séparer les caractéristiques et les étiquettes
	x = train_df[INCLUDE_FEATURES]
	y = train_df["diagnosis"].map({"M": 1, "B": 0})

	# Preprocessing
	x_mean = []
	x_std = []
	n_features = len(INCLUDE_FEATURES)

	for feature in INCLUDE_FEATURES:
		feature_values = x[feature].tolist()
		x_mean.append(st.ft_mean(feature_values))
		x_std.append(st.ft_std_dev(feature_values))

	x_norm = []
	for patient in range(len(x)):
		normed_features = []
		for feature in range(n_features):
			val = x.iloc[patient, feature]
			if val is None or (isinstance(val, (int, float)) and not (val != val)):
				val = x_mean[feature]
			std = x_std[feature] if x_std[feature] != 0 else 1.0
			normed_features.append((val - x_mean[feature]) / std)
		x_norm.append(normed_features)

	x_norm = np.array(x_norm, dtype=float)
	y = np.array(y.tolist(), dtype=float)
	weights = np.random.uniform(-0.5, 0.5, size=n_features)
	bias = np.random.uniform(-0.5, 0.5)

	n = neurons(x_norm[0], weights=weights, bias=bias)
	print(n)

if __name__ == "__main__":
	main()