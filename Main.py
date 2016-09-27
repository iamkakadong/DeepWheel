from HW1.NetworkStructure import Network
import numpy as np
import matplotlib.pyplot as plt
import pickle
import multiprocessing
import os.path
from HW1.Helper import String

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

def cross_validation():
	cv_res = list()
	params = list()
	for learning_rate in [0.01, 0.1, 0.5]:
		for momentum in [0, 0.5]:
			for hidden_unit in [100, 200, 500]:
				for epochs in [50, 150, 250]:
					for regularization in [0, 0.1, 0.5]:
						for dropout_rate in [0, 0.5]:
							param = dict()
							param['learning_rate'] = learning_rate
							param['momentum'] = momentum
							param['hidden_unit'] = hidden_unit
							param['epochs'] = epochs
							param['regularziation'] = regularization
							param['dropout_rate'] = dropout_rate
							param['cv_res'] = cv_res
							params.append(param)
	subroutine(params[0])
	# pool = multiprocessing.Pool(processes=2)
	# cv_res = pool.map(subroutine, params)
	# pool.close()
	# pool.join()

	return cv_res

def subroutine(param):
	myNet = Network.Network()
	learning_rate = param['learning_rate']
	momentum = param['momentum']
	hidden_unit = param['hidden_unit']
	epochs = param['epochs']
	regularization = param['regularziation']
	dropout_rate = param['dropout_rate']

	# cv_res = param['cv_res']

	myNet.setLayer(3, [[784, "Input"], [hidden_unit, "Sigmoid"], [10, "Softmax"]])
	myNet.setLearningRate(learning_rate)
	myNet.setMomentum(momentum)
	myNet.setL2Weight(regularization)
	myNet.setDropOutRate(dropout_rate)
	myNet.initNetwork()

	name = myNet.getName() + '_ep_' + String.sciFormat(epochs)
	if not os.path.isfile('Results/ce_loss_' + name + '.png'):
		[loss, closs, vloss, vcloss] = myNet.trainAndValidate(X_train, y_train, X_val, y_val, epochs)

		fig = plotError(myNet, loss, vloss, "Cross-entropy Loss")
		fig.savefig('Results/ce_loss_' + name + '.png', format='png')
		fig = plotError(myNet, closs, vcloss, "Misclassification Error")
		fig.savefig('Results/mc_loss_' + name + '.png', format='png')
		print "cross-validation result for " + name + ":\n"
		print "\ttraining cross-entropy loss: %0.3f\n" % loss
		print "\ttraining error rate: %0.3f\n" % closs
		print "\tcross-validation cross-entropy loss: %0.3f\n" % vloss
		print "\tcross-validation error rate: %0.3f\n" % vcloss
		# cv_res.append([loss,closs,vloss,vcloss,name])
		return [loss,closs,vloss,vcloss,name]

if __name__ == '__main__':
	[X_train, y_train] = loadData("HW1/data/digitstrain.txt")
	[X_val, y_val] = loadData("HW1/data/digitsvalid.txt")

	# myNet = Network.Network()
	cv_res = cross_validation()
	with open('cv_result', 'wb') as f:
		pickle.dump(cv_res, f)

	# myNet.setLayer(3, [[784, "Input"], [500, "Sigmoid"], [10, "Softmax"]])
	# myNet.setLearningRate(0.01)
	# myNet.initNetwork()
	# myNet.setMomentum(0.5)
	# myNet.setL2Weight(0.0)
	# epochs = 200
	#
	# [loss, closs, vloss, vcloss] = myNet.trainAndValidate(X_train, y_train, X_val, y_val, epochs)
	#
	# fig = plotError(myNet, loss, vloss, "Cross-entropy Loss")
	# fig.savefig('../HW1/Results/ce_loss_lr1e-2_mo5e-1_hl5e+2.png', format='png')
	# fig = plotError(myNet, closs, vcloss, "Misclassification Error")
	# fig.savefig('../HW1/Results/mc_loss_lr1e-2_mo5e-1_hl5e+2.png', format='png')
	# # fig = myNet.visualizeLayer(1)
	# # fig.savefig('../HW1/Results/visualize_weight.png', format='png')

