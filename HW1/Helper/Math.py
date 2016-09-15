from numpy import exp

def softmax(a):
	"""

	:rtype: ndarray
	"""
	t1 = exp(a)
	return t1 / t1.sum()

def sigmoid(a):
	return 1 / (1 + exp(-a))
