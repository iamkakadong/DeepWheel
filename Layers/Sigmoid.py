import numpy as np

from Helper import Math
from Layer import Layer


class Sigmoid(Layer):

	def top_layer_grad(self, truth):
		assert len(truth) == self.n_out
		grad = (self.d_out - truth) / (self.d_out * (1 - self.d_out))
		return grad

	def back_prop(self, post_activation_gradient):
		"""

		:type post_activation_gradient: np.ndarray
		"""
		assert len(post_activation_gradient) == self.n_out
		pre_activation_gradient = post_activation_gradient * self.d_out * (1 - self.d_out)
		weight_gradient = self.weight_gradient * self.momentum + np.outer(pre_activation_gradient, self.d_in)
		in_post_gradient = np.dot(self.weights[:, 0: -1].transpose(), pre_activation_gradient)
		self.gradient = in_post_gradient
		self.weight_gradient = weight_gradient

	def feed_forward(self):
		assert self.has_in()
		self.d_out = Math.sigmoid(np.dot(self.weights, self.d_in))

	def __init__(self, n_in, n_out, momentum=0.0, reg_param=0.0, dropout=0.0):
		Layer.__init__(self, n_in, n_out, momentum, reg_param, dropout)
