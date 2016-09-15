from ..Layers import Sigmoid
import numpy as np

class Network:
	n_layers = 0
	n_units = []
	learning_rate = 0.0
	dropout_rate = 0.0
	l2_weight = 0.0
	momentum = 0.0
	layers = list()
	n_features = 0

	def initNetwork(self):
		# Check network is correctly specified
		if (not self.isValidNetwork()): return

		layers = list()
		for i in range(self.n_layers):
			tmp = list()
			tmp.append(self.n_features)
			tmp.extend(self.n_units)
			layers.append(Sigmoid.Sigmoid(tmp[i], tmp[i + 1]))

		self.layers = layers

	def forwardProp(self, input):
		"""

		:type input: np.ndarray
		"""
		current_output = input
		for layer in self.layers:
			assert isinstance(layer, Sigmoid.Sigmoid)
			layer.setInput(current_output)
			layer.feedForward()
			current_output = layer.d_out

	def getLoss(self, truth, function):
		assert len(self.layers) > 0
		return function(self.layers[-1].d_out, truth)

	def backProp(self, truth):
		output_layer = self.layers[-1]
		assert isinstance(output_layer, Sigmoid.Sigmoid)
		grad = np.zeros(self.n_units[-1])
		grad[np.where(truth == 1)] = -1
		grad += output_layer.d_out
		for layer in self.layers[-1:0:-1]:
			assert isinstance(layer, Sigmoid.Sigmoid)
			layer.backProp(grad)
			layer.updateWeight(self.learning_rate)
			grad = layer.gradient

	def setLayer(self, n, units):
		assert len(units) == n
		self.n_layers = n
		self.n_units = units

	def setLearningRate(self, rate):
		self.learning_rate = rate

	def setDropOutRate(self, rate):
		self.dropout_rate = rate

	def setL2Weight(self, weight):
		self.l2_weight = weight

	def setMomentum(self, momentum):
		self.momentum = momentum

	def setFeatures(self, n_features):
		self.n_features = n_features

	def isValidNetwork(self):
		if self.n_layers == 0:
			print 'This network is empty. i.e., number of layer is 0'
			return False
		if self.n_layers != len(self.n_units):
			print 'Misspecified network: number of layers does not match with units array'
			return False
		if self.learning_rate == 0:
			print 'Learning rate is 0'
			return False
		return True

	def __init__(self):
		return