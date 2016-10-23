import numpy as np

def load_binary_data(filename):
	x = list()
	y = list()
	f = file(filename)
	for line in f.readlines():
		nums = map(lambda e: float(e), line.split(','))
		[xt, yt] = to_binary_xy(nums)
		x.append(xt)
		y.append(yt)
	return [x, y]

def load_data(filename):
	x = list()
	y = list()
	f = file(filename)
	for line in f.readlines():
		nums = map(lambda e: float(e), line.split(','))
		[xt, yt] = to_xy(nums)
		x.append(xt)
		y.append(yt)
	return [x, y]

def to_xy(array):
	x = np.array(array[0:-1])
	label = int(array[-1])
	y = np.zeros(10)
	y[label] = 1
	return x, y

def to_binary_xy(array):
	x = (np.array(array[0:-1]) > 0.5).astype(int)
	label = int(array[-1])
	y = np.zeros(10)
	y[label] = 1
	return x, y
