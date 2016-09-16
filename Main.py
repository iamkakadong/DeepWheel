from HW1.NetworkStructure import Network
import numpy as np

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

if __name__ == '__main__':
	[X_train, y_train] = loadData("/Users/tianshuren/Google Drive/2016 Fall/10807/Code/HW1/data/digitstrain.txt")
	[X_val, y_val] = loadData("/Users/tianshuren/Google Drive/2016 Fall/10807/Code/HW1/data/digitsvalid.txt")

	myNet = Network.Network()
	myNet.setFeatures(784)
	myNet.setLayer(2, [[100, "Sigmoid"], [10, "Softmax"]])
	myNet.setLearningRate(0.1)
	myNet.initNetwork()
	myNet.setMomentum(0.0)
	myNet.setL2Weight(0.0)
	epochs = 200

	# [X_train, y_train] = [map(lambda a : toXY(a)[0], d_train), map(lambda a : toXY(a)[1], d_train)]
	# [X_val, y_val] = [map(lambda a : toXY(a)[0], d_validation), map(lambda a : toXY(a)[1], d_validation)]
	[loss, closs, vloss, vcloss] = myNet.trainAndValidate(X_train, y_train, X_val, y_val, epochs)
	# loss = []
	# closs = []
	# cv_loss = []
	# cv_closs = []
	#
	# for epoch in range(epochs):
	# 	loss_iter = []
	# 	closs_iter = []
	# 	for entry in np.random.permutation(d_train):
	# 		(X, y) = toXY(entry)
	# 		myNet.forwardProp(X)
	# 		loss_iter.append(myNet.getLoss(y, LossFun.crossEntropy))
	# 		closs_iter.append(myNet.getLoss(y, LossFun.classificationError))
	# 		myNet.backProp(y)
	# 	loss.append(np.mean(loss_iter))
	# 	closs.append(np.mean(closs_iter))
	#
	# 	print 'training loss (Cross-Entropy) in iteration %d is: %0.3f' % (epoch, loss[-1])
	# 	print 'training loss (Accuracy) in iteration %d is: %0.3f' % (epoch, closs[-1])
	#
	# 	cv_loss_iter = []
	# 	cv_closs_iter = []
	# 	for entry in d_validation:
	# 		(X, y) = toXY(entry)
	# 		myNet.forwardProp(X)
	# 		cv_loss_iter.append(myNet.getLoss(y, LossFun.crossEntropy))
	# 		cv_closs_iter.append(myNet.getLoss(y, LossFun.classificationError))
	# 	cv_loss.append(np.mean(cv_loss_iter))
	# 	cv_closs.append(np.mean(cv_closs_iter))
	# 	print 'cross-validation loss (Cross-Entropy) in iteration %d is: %0.3f' % (epoch, cv_loss[-1])
	# 	print 'cross-validation loss (Accuracy) in iteration %d is: %0.3f' % (epoch, cv_closs[-1])
