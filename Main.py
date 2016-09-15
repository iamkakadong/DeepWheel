from HW1.NetworkStructure import Network
from HW1.Helper import LossFun
import numpy as np

def loadData(filename):
	data = list()
	f = file(filename)
	for line in f.readlines():
		data.append(map(lambda x : float(x), line.split(',')))
	return data

def toXY(array):
	X = np.array(array[0:-1])
	label = int(array[-1])
	y = np.zeros(10)
	y[label] = 1
	return X, y

if __name__ == '__main__':
	d_train = loadData("/Users/tianshuren/Google Drive/2016 Fall/10807/Code/HW1/data/digitstrain.txt")
	d_validation = loadData("/Users/tianshuren/Google Drive/2016 Fall/10807/Code/HW1/data/digitsvalid.txt")

	myNet = Network.Network()
	myNet.setFeatures(784)
	myNet.setLayer(2, [100, 10])
	myNet.setLearningRate(0.1)
	myNet.initNetwork()
	epochs = 200

	loss = []
	closs = []
	cv_loss = []
	cv_closs = []

	for epoch in range(epochs):
		loss_iter = []
		closs_iter = []
		for entry in d_train:
			(X, y) = toXY(entry)
			myNet.forwardProp(X)
			single_loss = myNet.getLoss(y, LossFun.crossEntropy)
			single_loss_c = myNet.getLoss(y, LossFun.classificationError)
			myNet.backProp(y)
			loss_iter.append(single_loss)
			closs_iter.append(single_loss_c)
		loss.append(np.mean(loss_iter))
		closs.append(np.mean(closs_iter))

		print 'training loss (Cross-Entropy) in iteration %d is: %0.3f' % (epoch, loss[-1])
		print 'training loss (Accuracy) in iteration %d is: %0.3f' % (epoch, closs[-1])

		cv_loss_iter = []
		cv_closs_iter = []
		for entry in d_train:
			(X, y) = toXY(entry)
			myNet.forwardProp(X)
			cv_single_loss = myNet.getLoss(y, LossFun.crossEntropy)
			cv_loss_iter.append(cv_single_loss)
			cv_closs_iter.append(myNet.getLoss(y, LossFun.classificationError))
		cv_loss.append(np.mean(cv_loss_iter))
		cv_closs.append(np.mean(cv_closs_iter))
		print 'cross-validation loss (Cross-Entropy) in iteration %d is: %0.3f' % (epoch, cv_loss[-1])
		print 'cross-validation loss (Accuracy) in iteration %d is: %0.3f' % (epoch, cv_closs[-1])
