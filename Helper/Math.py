import numpy as np

def softmax(a):
	"""

	:rtype: ndarray
	"""

	e_x = np.exp(a - np.max(a))
	out = e_x / e_x.sum()
	return out

def sigmoid(a):
	return 1 / (1 + np.exp(-a))
