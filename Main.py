from HW1.NetworkStructure import Network
import numpy as np
import matplotlib.pyplot as plt

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

def plotError(network, training_loss, cv_loss, loss_label):
	fig = plt.figure()
	plt.plot(training_loss, label = 'Training loss')
	plt.hold(True)
	plt.plot(cv_loss, label = 'Cross-validation loss')
	plt.xlabel('Epoch Number')
	plt.ylabel(loss_label)
	plt.title(network.printStruct())
	plt.legend()
	return fig

if __name__ == '__main__':
	[X_train, y_train] = loadData("/Users/tianshuren/Google Drive/2016 Fall/10807/Code/HW1/data/digitstrain.txt")
	[X_val, y_val] = loadData("/Users/tianshuren/Google Drive/2016 Fall/10807/Code/HW1/data/digitsvalid.txt")

	myNet = Network.Network()
	myNet.setLayer(3, [[784, "Input"], [500, "Sigmoid"], [10, "Softmax"]])
	myNet.setLearningRate(0.01)
	myNet.initNetwork()
	myNet.setMomentum(0.5)
	myNet.setL2Weight(0.0)
	epochs = 200

	[loss, closs, vloss, vcloss] = myNet.trainAndValidate(X_train, y_train, X_val, y_val, epochs)

	fig = plotError(myNet, loss, vloss, "Cross-entropy Loss")
	fig.savefig('../HW1/Results/ce_loss_lr1e-2_mo5e-1_hl5e+2.png', format='png')
	fig = plotError(myNet, closs, vcloss, "Misclassification Error")
	fig.savefig('../HW1/Results/mc_loss_lr1e-2_mo5e-1_hl5e+2.png', format='png')
	# fig = myNet.visualizeLayer(1)
	# fig.savefig('../HW1/Results/visualize_weight.png', format='png')
