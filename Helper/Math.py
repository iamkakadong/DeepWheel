import numpy as np

def softmax(a):
	"""

	:type a: np.ndarray
	:rtype: np.ndarray
	"""

	e_x = np.exp(a - np.max(a))
	out = e_x / e_x.sum()
	return out

def sigmoid(a):
	return 1 / (1 + np.exp(-a))
