import numpy as np
from ..Helper import RandomInit, Math
from abc import ABCMeta, abstractmethod

class Layer:
	__metaclass__ = ABCMeta

	n_in = 0
	n_out = 0
	weights = np.ndarray([0])  # a matrix of size n_out * n_in
	d_out = np.ndarray([0])
	d_in = np.ndarray([0])
	gradient = np.ndarray([0])
	weight_gradient = np.ndarray([0])
	momentum = 0.0
	reg_param = 0.0
	dropout = 0.0

	@abstractmethod
	def feedForward(self):
		pass

	def setInput(self, d_in):
		assert isinstance(d_in, np.ndarray)
		self.d_in = np.append(d_in, [1]) # add the bias term

	@abstractmethod
	def backProp(self, post_activation_gradient):
		"""

		:type post_activation_gradient: np.ndarray
		"""
		pass

	def updateWeight(self, learning_rate):
		self.weights -= learning_rate * self.weight_gradient

	def hasIn(self):
		return self.d_in != None

	def __init__(self, n_in, n_out, momentum, reg_param, dropout):
		self.n_in = n_in
		self.n_out = n_out
		self.weights = RandomInit.uniformInit([n_out, n_in + 1], -1, 1)
		self.weight_gradient = np.zeros([n_out, n_in + 1])
		self.momentum = momentum
		self.reg_param = reg_param
		self.dropout = dropout