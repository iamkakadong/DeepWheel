from Layer import Layer
from ..Helper import Math
import numpy as np


class Softmax(Layer):

	def feedForward(self):
		assert self.hasIn()
		self.d_out = Math.softmax(self.weights.dot(self.d_in))

	def backProp(self, post_activation_gradient):
		# TODO: Implement momentum, regularization
		assert len(post_activation_gradient) == self.n_out
		t1 = self.d_out
		t2 = - np.outer(t1, t1) + np.diag(t1)
		pre_activation_gradient = np.dot(post_activation_gradient, t2)
		weight_gradient = np.outer(pre_activation_gradient, self.d_in)
		in_post_gradient = np.dot(self.weights[:, 0 : -1].transpose(), pre_activation_gradient)
		self.gradient = in_post_gradient
		self.weight_gradient = weight_gradient

	def __init__(self, n_in, n_out, momentum = 0.0, reg_param = 0.0, dropout = 0.0):
		Layer.__init__(self, n_in, n_out, momentum, reg_param, dropout)