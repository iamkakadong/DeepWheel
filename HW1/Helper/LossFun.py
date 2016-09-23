from numpy import where, log


def crossEntropy(predict, truth):
	validateClassification(predict, truth)
	return -log(predict[where(truth == 1)])

def classificationError(predict, truth):
	validateClassification(predict, truth)
	plabel = where(predict == max(predict))
	tlabel = where(truth == 1)
	return plabel != tlabel

def validateClassification(predict, truth):
	assert sum(truth == 1) == 1 and sum(truth == 0) == len(truth) - 1
	assert len(predict) == len(truth)