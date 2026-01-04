class Network:
	def __init__(self):
		self.layers = []

	def forward(self, x):
		raise NotImplementedError("Forward method not implemented.")

	def backward(self, y_true, learning_rate):
		raise NotImplementedError("Backward method not implemented.")

class SequentialNetwork(Network):
	def __init__(self, layers):
		self.layers = layers

	def forward(self, x):
		# Copy the provided layers into the network to avoid index errors
		for layer in self.layers:
			x = layer.forward(x)
		return x

	def backward(self, y_true, learning_rate):
		pass

	def fit(self, network, data_train, data_valid, loss="None", learning_rate=0.0, batch_size=1, epochs=1):
		pass