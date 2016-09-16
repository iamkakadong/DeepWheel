from ..Layers.Sigmoid import Sigmoid
from ..Layers.Softmax import Softmax
from ..Layers.Layer import Layer
from ..Helper import LossFun
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
		if (not self.isValidNetwork()):
			return

		layers = list()
		in_size = self.n_features
		for u in self.n_units:
			out_size = u[0]
			type = u[1]
			if (type == "Sigmoid"):
				layers.append(Sigmoid(in_size, out_size, self.momentum, self.l2_weight))
			elif (type == "Softmax"):
				layers.append(Softmax(in_size, out_size, self.momentum, self.l2_weight))
			else:
				print "invalid layer type: " + type
				return
			in_size = out_size

		self.layers = layers

	def reset(self):
		self.initNetwork()

	def trainAndValidate(self, X_train, y_train, X_val, y_val, epoch):
		cel_train = list()
		cel_val = list()
		acc_train = list()
		acc_val = list()
		for i in range(epoch):
			print "epoch # %d: " % i
			print "Training..."
			[ct, at] = self.train(X_train, y_train, 1)
			cel_train.extend(ct)
			acc_train.extend(at)
			print "    CE: %0.3f; ACC: %0.3f" % (ct[0], at[0])
			print "Validating..."
			[ct, at] = self.validate(X_val, y_val)
			cel_val.append(ct)
			acc_val.append(at)
			print "    CE: %0.3f; ACC: %0.3f" % (ct, at)
		return [cel_train, acc_train, cel_val, acc_val]

	def validate(self, X, y):
		assert len(X) == len(y)
		cel = list()
		acc = list()
		for i in range(len(y)):
			self.forwardProp(X[i])
			cel.append(self.getLoss(y[i], LossFun.crossEntropy))
			acc.append(self.getLoss(y[i], LossFun.classificationError))
		return [np.mean(cel), np.mean(acc)]

	def train(self, X, y, epoch):
		cel = list()
		acc = list()
		for i in range(epoch):
			cel_accumulator = list()
			acc_accumulator = list()
			for j in np.random.permutation(range(len(y))):
				self.forwardProp(X[j])
				cel_accumulator.append(self.getLoss(y[j], LossFun.crossEntropy))
				acc_accumulator.append(self.getLoss(y[j], LossFun.classificationError))
				self.backProp(y[j])
			cel.append(np.mean(cel_accumulator))
			acc.append(np.mean(acc_accumulator))
		return cel, acc

	def forwardProp(self, input):
		"""

		:type input: np.ndarray
		"""
		current_output = input
		for layer in self.layers:
			assert isinstance(layer, Layer)
			layer.setInput(current_output)
			layer.feedForward()
			current_output = layer.d_out

	def getLoss(self, truth, function):
		assert len(self.layers) > 0
		return function(self.layers[-1].d_out, truth)

	def backProp(self, truth):
		output_layer = self.layers[-1]
		assert isinstance(output_layer, Layer)
		grad = np.zeros(self.n_units[-1][0])
		label = np.where(truth == 1)
		grad[label] = - 1.0 / output_layer.d_out[label]
		for layer in self.layers[::-1]:
			assert isinstance(layer, Layer)
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