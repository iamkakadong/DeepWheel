from numpy import where, log, linalg

def seLoss(predict, truth):
	return linalg.norm(predict - truth)

def crossEntropy(predict, truth):
	# validateClassification(predict, truth)
	validateBinaryPrediction(predict, truth)
	return -log(predict[where(truth == 1)])

def classificationError(predict, truth):
	# validateClassification(predict, truth)
	"""

	:type predict: numpy.ndarray
	"""
	validateBinaryPrediction(predict, truth)
	return sum((predict == truth).astype(int))
	# plabel = where(predict == max(predict))
	# tlabel = where(truth == 1)
	# return plabel != tlabel

def validateClassification(predict, truth):
	assert sum(truth == 1) == 1 and sum(truth == 0) == len(truth) - 1
	assert len(predict) == len(truth)

def validateBinaryPrediction(predict, truth):
	assert len(predict) == len(truth)
	assert sum(truth == 1) + sum(truth == 0) == len(truth)