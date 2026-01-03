import pandas as pd
import numpy as np
from utils import stats_tools as st

INCLUDE_FEATURES = ["texture_mean", "perimeter_mean", "compactness_mean", "concavity_mean","concave_points_mean"]


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

class DenseLayer:
	def __init__(self, units, activation, weights_initializer="random"):
		self.units = units
		self.activation = activation
		self.weights_initializer = weights_initializer

		self.weights = None
		self.bias = None

	def initialize(self, input_size):
		if self.weights_initializer == "heUniform":
			limit = np.sqrt(6 / input_size)
			self.weights = np.random.uniform(-limit, limit, (input_size, self.units))
		else:
			self.weights = np.random.randn(input_size, self.units)
		self.bias = np.zeros((1, self.units))

	def activate(self, x): 
		if self.activation == "sigmoid":
			return 1 / (1 + np.exp(-x))
		elif self.activation == "softmax":
			exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
			return exp_x / exp_x.sum(axis=1, keepdims=True)
		else:
			raise ValueError("Unsupported activation function")

class MLP:
	def __init__(self):
		self.layers = []

	def CreateNetwork(self, layers_config):
		self.layers.append(layers_config[0])
		self.layers[0].initialize(input_size=layers_config[0].units)
		for i in range(1, len(layers_config)):
			self.layers[i] = layers_config[i].initialize(input_size=self.layers[i-1].units)

	def fit(self, network, data_train, data_valid, loss="None", learning_rate=0.0, batch_size=1, epochs=1):
		pass



def neurons(x, weights, bias):
	return sigmoide(sum(x[i] * weights[i] for i in range(len(x))) + bias)

def loss(y_true, y_pred):
	n= len(y_true)
	return -sum(y_true[i] * np.log(y_pred[i]) + (1 - y_true[i]) * np.log(1 - y_pred[i]) for i in range(n)) / n

def prepocess(data, x_mean=None, x_std=None):
	# Séparer les caractéristiques et les étiquettes
	x = data[INCLUDE_FEATURES]
	y = data["diagnosis"].map({"M": 1, "B": 0})

	# Preprocessing
	n_features = len(INCLUDE_FEATURES)

	if x_mean is None or x_std is None:
		x_mean = []
		x_std = []
		for feature in INCLUDE_FEATURES:
			feature_values = x[feature].tolist()
			x_mean.append(st.ft_mean(feature_values))
			x_std.append(st.ft_std_dev(feature_values))

	x_norm = []
	for patient in range(len(x)):
		normed_features = []
		for feature in range(n_features):
			val = x.iloc[patient, feature]
			if np.isnan(val):
				val = x_mean[feature]
			std = x_std[feature] if x_std[feature] != 0 else 1.0
			normed_features.append((val - x_mean[feature]) / std)
		x_norm.append(normed_features)

	x_norm = np.array(x_norm, dtype=float)
	y = np.array(y.tolist(), dtype=float)
	return (x_norm, y, x_mean, x_std)

def main(argv=None):
	#! check if data file exists
	df = pd.read_csv("data_splits/train_data.csv")

	df = df.sample(frac=1, random_state=42).reset_index(drop=True)
	split = int(len(df) * 0.2)
	data_train = df.iloc[split:]
	data_valid = df.iloc[:split]
	
	x_mean = []
	x_std = []

	x_train, y_train, x_mean, x_std = prepocess(data_train)
	x_valid, y_valid, _, _ = prepocess(data_valid, x_mean, x_std)

	data_train = (x_train, y_train)
	data_valid = (x_valid, y_valid)

	model = MLP()

	network_config = model.CreateNetwork([
		DenseLayer(units=10, activation="sigmoid", weights_initializer="heUniform"),
		DenseLayer(units=5, activation="sigmoid", weights_initializer="heUniform"),
		DenseLayer(units=2, activation="softmax", weights_initializer="random")
	])




if __name__ == "__main__":
	main()