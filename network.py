import numpy as np

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

	def backward(self, output_gradient, learning_rate):
		grad = output_gradient
		for layer in reversed(self.layers):
			grad = layer.backward(grad, learning_rate)
		return grad


	def fit(self, network, data_train, data_valid, loss="None", learning_rate=0.0, batch_size=1, epochs=1):
		for epoch in range(epochs):
			train_loss = 0.0
			x_train, y_train = data_train
			for x, y in zip(x_train, y_train):
				output = network.forward(x.reshape(1, -1))
				# One-hot encode target for 2-class output
				y_one_hot = np.zeros_like(output)
				y_one_hot[0, int(y)] = 1
				loss_value = loss.forward(output, y_one_hot)
				train_loss += float(np.mean(loss_value))
				output_gradient = loss.backward(output, y_one_hot)
				network.backward(output_gradient, learning_rate)
			train_loss /= len(x_train)

			val_loss = 0.0
			val_correct = 0
			x_valid, y_valid = data_valid
			for x_val, y_val in zip(x_valid, y_valid):
				val_output = network.forward(x_val.reshape(1, -1))
				val_y_one_hot = np.zeros_like(val_output)
				val_y_one_hot[0, int(y_val)] = 1
				val_loss += float(np.mean(loss.forward(val_output, val_y_one_hot)))
				pred_class = int(np.argmax(val_output, axis=1)[0])
				if pred_class == int(y_val):
					val_correct += 1

			val_loss /= len(x_valid)
			val_accuracy = val_correct / len(x_valid)
			print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")