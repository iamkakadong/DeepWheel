import numpy as np

def square_error(predict, truth):
	return np.linalg.norm(predict - truth)

def cross_entropy(predict, truth):
	return sum(-np.log(predict[np.where(truth == 1)]))

def misclassify_rate(predict, truth):
	"""

	:type truth: np.ndarray
	:type predict: np.ndarray
	"""
	p_label = np.where(predict == max(predict))
	t_label = np.where(truth == 1)
	return p_label != t_label

def validate_classification(predict, truth):
	assert sum(truth == 1) == 1 and sum(truth == 0) == len(truth) - 1
	assert len(predict) == len(truth)

def validate_binary_prediction(predict, truth):
	assert len(predict) == len(truth)
	assert sum(truth == 1) + sum(truth == 0) == len(truth)
