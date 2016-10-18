import numpy as np

def loadBinaryData(filename):
	X = list()
	y = list()
	f = file(filename)
	for line in f.readlines():
		nums = map(lambda x : float(x), line.split(','))
		[xt, yt] = toBinaryXY(nums)
		X.append(xt)
		y.append(yt)
	return [X, y]

def loadData(filename):
	X = list()
	y = list()
	f = file(filename)
	for line in f.readlines():
		nums = map(lambda x : float(x), line.split(','))
		[xt, yt] = toXY(nums)
		X.append(xt)
		y.append(yt)
	return [X, y]

def toXY(array):
	X = np.array(array[0:-1])
	label = int(array[-1])
	y = np.zeros(10)
	y[label] = 1
	return X, y

def toBinaryXY(array):
	X = (np.array(array[0:-1]) > 0.5).astype(int)
	label = int(array[-1])
	y = np.zeros(10)
	y[label] = 1
	return X, y