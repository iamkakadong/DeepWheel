import matplotlib.pyplot as plt
import numpy as np

from Helper import String, Pair

from Layers.Layer import Layer

from abc import ABCMeta

class Network:
	__metaclass__ = ABCMeta

	n_layers = 0
	n_units = []
	learning_rate = 0.0
	output = []
	dropout_rate = 0.0
	l2_reg = 0.0
	momentum = 0.0
	layers = list()
	lf_name = list()
	lf_ptr = list()
	tied_weight_pair = Pair.Pair()

	def unpack_loss(self, loss, accumulator):
		m_loss = map(lambda x: np.mean(x), loss)
		n_lf = len(loss)
		msg = "   "
		for k in range(n_lf):
			msg += ' ' + self.lf_name[k] + u': {0:0.3f};'.format(m_loss[k])
			accumulator[k].append(m_loss[k])
		print msg

	def train_and_validate(self, x_train, y_train, x_val, y_val, epoch):
		loss_train = list()
		loss_valid = list()
		for i in range(len(self.lf_name)):
			loss_train.append(list())
			loss_valid.append(list())
		for i in range(epoch):
			print "epoch # %d: " % i
			print "Validating..."
			loss = self.validate(x_val, y_val)
			self.unpack_loss(loss, loss_valid)
			print "Training..."
			self.train(x_train, y_train, 1)
			loss = self.validate(x_train, y_train)
			self.unpack_loss(loss, loss_train)
		return loss_train, loss_valid

	def validate(self, x, y):
		assert len(x) == len(y)
		n_lf = len(self.lf_name)
		loss = list()
		for i in range(n_lf):
			loss.append(list())
		for i in range(len(y)):
			self.forward_prop(x[i])
			for k in range(n_lf):
				loss[k].append(self.get_loss(y[i], self.lf_ptr[k]))
		return loss

	def train(self, x, y, epoch):
		# TODO: this epoch may not be necessary
		for i in range(epoch):
			for j in np.random.permutation(range(len(y))):
				self.forward_prop_train(x[j])
				self.back_prop(y[j])

	def forward_prop_train(self, x_in):
		"""

		:type x_in: np.ndarray
		"""
		current_output = x_in
		for layer in self.layers:
			assert isinstance(layer, Layer)
			layer.set_input(current_output)
			layer.feed_forward_train()
			current_output = layer.d_out

	def forward_prop(self, x_in):
		current_output = x_in
		for layer in self.layers:
			assert isinstance(layer, Layer)
			layer.set_input(current_output)
			layer.feed_forward_test()
			current_output = layer.d_out
		self.output = current_output

	def back_prop(self, truth):
		output_layer = self.layers[-1]
		assert isinstance(output_layer, Layer)
		grad = output_layer.top_layer_grad(truth)
		for layer in self.layers[::-1]:
			assert isinstance(layer, Layer)
			layer.back_prop(grad)
			layer.update_weight(self.learning_rate)
			if layer in self.tied_weight_pair:
				g_tmp = layer.weight_gradient[:, 0:-1].transpose()
				[n_out, n_in] = g_tmp.shape
				g_tmp = np.c_[g_tmp, np.zeros([n_out, 1])]
				l_other = self.tied_weight_pair.other(layer)
				assert isinstance(l_other, Layer)
				l_other.weight_gradient = g_tmp
				l_other.update_weight(self.learning_rate)
			grad = layer.gradient

	def get_loss(self, truth, function):
		assert len(self.layers) > 0
		return function(self.layers[-1].d_out, truth)

	def set_layer(self, layers, units):
		self.layers = layers
		self.n_layers = len(layers)
		self.n_units = units

	def set_learning_rate(self, rate):
		self.learning_rate = rate

	def set_loss_func(self, name, functions):
		self.lf_name = name
		self.lf_ptr = functions

	def set_dropout_rate(self, dropout_rate):
		self.dropout_rate = dropout_rate
		for layer in self.layers:
			layer.dropout_rate = dropout_rate

	def set_l2_reg(self, l2_reg):
		self.l2_reg = l2_reg
		for layer in self.layers:
			layer.reg_param = l2_reg

	def set_momentum(self, momentum):
		self.momentum = momentum
		for layer in self.layers:
			layer.momentum = momentum

	def set_weight_pair(self, i1, i2):
		l1 = self.layers[i1]
		l2 = self.layers[i2]
		self.tied_weight_pair.setPair(l1, l2)
		assert isinstance(l1, Layer)
		assert isinstance(l2, Layer)
		tmp = l1.weights[:, 0:-1].transpose()
		[n_out, n_in] = tmp.shape
		tmp = np.c_[tmp, np.zeros([n_out, 1])]
		l2.weights = tmp

	def is_valid_network(self):
		if self.n_layers == 0:
			print 'This network is empty. i.e., number of layer is 0'
			return False
		if self.n_layers != len(self.n_units):
			print 'Ill specified network: number of layers does not match with units array'
			return False
		if self.learning_rate == 0:
			print 'Learning rate is 0'
			return False
		return True

	def get_layer_struct(self):
		s = "["
		for n_unit in self.n_units[:-1]:
			s += str(n_unit) + "->"
		s += str(self.n_units[-1]) + "]"
		return s

	def get_name(self):
		if len(self.n_units) == 0:
			print 'Network is Empty!'
			return
		# return "lr_" + String.sciFormat(self.learning_rate) + "_mo_" + String.sciFormat(self.momentum) + \
		# 	   "_hu_" + self.getLayerStruct() +"_l2_" + String.sciFormat(self.l2_weight) + "_dr_" + \
		# 		String.sciFormat(self.dropout_rate)
		return "lr_" + String.sci_format(self.learning_rate) + "_hu_" + self.get_layer_struct()

	def visualize_layer(self, layer_idx):
		# Visualize i-th layer
		assert layer_idx != 0
		tmp = self.layers[layer_idx - 1].weights[:, :-1]
		size = int(np.ceil(np.sqrt(self.n_units[layer_idx][0])))
		f, axarr = plt.subplots(size, size)
		for i in range(size):
			for j in range(size):
				idx = i * size + j
				if idx >= tmp.shape[0]:
					break
				vis_dim = int(np.sqrt(self.n_units[layer_idx - 1][0]))
				axarr[i, j].matshow(np.reshape(tmp[idx, :], [vis_dim, vis_dim]), cmap=plt.cm.gray)
				axarr[i, j].axis('off')
		return f

	def print_struct(self):
		s = self.get_layer_struct() + ";\n"
		s += ("lr %0.2f; " % self.learning_rate)
		# s += ("dr %0.2f; " % self.dropout_rate)
		# s += ("momentum %0.2f; " % self.momentum)
		# s += ("l2 %0.2f" % self.l2_weight)
		return s

	def reset_all(self):
		for layer in self.layers:
			layer.reset()

	def reset_layer(self, i):
		tmp = self.layers[i]
		assert isinstance(tmp, Layer)
		tmp.reset()

	def __init__(self):
		return
