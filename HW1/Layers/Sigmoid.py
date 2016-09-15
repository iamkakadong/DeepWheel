import numpy as np
from ..Helper import RandomInit, Math


class Sigmoid:
	n_in = 0
	n_out = 0
	weights = np.ndarray([0])  # a matrix of size n_out * n_in
	d_out = np.ndarray([0])
	d_in = np.ndarray([0])
	gradient = np.ndarray([0])
	weight_gradient = np.ndarray([0])

	def feedForward(self):
		assert self.hasIn()
		self.d_out = Math.softmax(self.weights.dot(self.d_in))

	def setInput(self, d_in):
		assert isinstance(d_in, np.ndarray)
		self.d_in = np.append(d_in, [1]) # add the bias term

	def backProp(self, pre_activation_gradient):
		"""

		:type pre_activation_gradient: np.ndarray
		"""
		assert len(pre_activation_gradient) == self.n_out
		weight_gradient = np.outer(pre_activation_gradient, self.d_in)
		in_post_gradient = np.dot(self.weights[:, 0 : -1].transpose(), pre_activation_gradient)
		t1 = self.d_in[0:-1]
		t2 = - np.outer(t1, t1) + np.eye(self.n_in).dot(t1)
		in_pre_gradient = np.dot(in_post_gradient, t2)
		self.gradient = in_pre_gradient
		self.weight_gradient = weight_gradient

	def updateWeight(self, learning_rate):
		self.weights -= learning_rate * self.weight_gradient

	def hasIn(self):
		return self.d_in != None

	def __init__(self, n_in, n_out):
		self.n_in = n_in
		self.n_out = n_out
		self.weights = RandomInit.uniformInit([n_out, n_in + 1], -1, 1)
