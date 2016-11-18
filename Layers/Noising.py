from Layer import Layer
import numpy as np

class Noising(Layer):
	is_binary = True

	def feed_forward_test(self):
		self.feed_forward()

	def update_weight(self, learning_rate):
		pass

	def top_layer_grad(self, truth):
		print 'This cannot be used as a top layer'
		pass

	def feed_forward(self):
		assert self.has_in()
		self.d_out = self.d_in[0:-1]

	def dropout(self):
		self.d_out = np.random.binomial(1, 1 - self.dropout_rate, self.n_out)

	def back_prop(self, post_activation_gradient):
		self.gradient = post_activation_gradient

	def __init__(self, size, rate, is_binary):
		Layer.__init__(self, size, size, 0, 0, dropout=rate)
		self.is_binary = is_binary
