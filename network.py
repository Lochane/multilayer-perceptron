class SequentialNetwork:
	def __init__(self, layers):
		self.layers = layers

	def forward(self, x):
		# Copy the provided layers into the network to avoid index errors
		for layer in self.layers:
			x = layer.forward(x)
		return x

	def fit(self, network, data_train, data_valid, loss="None", learning_rate=0.0, batch_size=1, epochs=1):
		pass