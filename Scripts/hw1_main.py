import os.path

import matplotlib.pyplot as plt
from traits.util.deprecated import deprecated

from NetworkStructure import Network
from Helper import Loader
from Helper import String

def plot_error(network, training_loss, cv_loss, loss_label):
	fig = plt.figure()
	plt.plot(training_loss, label='Training loss')
	plt.hold(True)
	plt.plot(cv_loss, label='Cross-validation loss')
	plt.xlabel('Epoch Number')
	plt.ylabel(loss_label)
	plt.title(network.print_struct())
	plt.legend()
	return fig

@deprecated
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

@deprecated
def subroutine(param):
	myNet = Network.Network()
	learning_rate = param['learning_rate']
	momentum = param['momentum']
	hidden_unit = param['hidden_unit']
	epochs = param['epochs']
	regularization = param['regularziation']
	dropout_rate = param['dropout_rate']

	# cv_res = param['cv_res']

	myNet.set_layer(3, [[784, "Input"], [hidden_unit, "Sigmoid"], [10, "Softmax"]])
	myNet.set_learning_rate(learning_rate)
	myNet.set_momentum(momentum)
	myNet.set_l2_reg(regularization)
	myNet.set_dropout_rate(dropout_rate)
	myNet.initNetwork()

	name = myNet.get_name() + '_ep_' + String.sci_format(epochs)
	if not os.path.isfile('Results/ce_loss_' + name + '.png'):
		[loss, closs, vloss, vcloss] = myNet.train_and_validate(X_train, y_train, X_val, y_val, epochs)

		fig = plot_error(myNet, loss, vloss, "Cross-entropy Loss")
		fig.savefig('Results/ce_loss_' + name + '.png', format='png')
		fig = plot_error(myNet, closs, vcloss, "Misclassification Error")
		fig.savefig('Results/mc_loss_' + name + '.png', format='png')
		print "cross-validation result for " + name + ":\n"
		print "\ttraining cross-entropy loss: %0.3f\n" % loss[-1]
		print "\ttraining error rate: %0.3f\n" % closs[-1]
		print "\tcross-validation cross-entropy loss: %0.3f\n" % vloss[-1]
		print "\tcross-validation error rate: %0.3f\n" % vcloss[-1]
		# cv_res.append([loss,closs,vloss,vcloss,name])
		return [loss,closs,vloss,vcloss,name]

if __name__ == '__main__':
	[X_train, y_train] = Loader.load_data("../data/digitstrain.txt")
	[X_val, y_val] = Loader.load_data("../data/digitsvalid.txt")
	[X_test, y_test] = Loader.load_data("../data/digitstest.txt")

	myNet = Network.Network()
	# cv_res = cross_validation()
	# with open('cv_result', 'wb') as f:
	# 	pickle.dump(cv_res, f)

	myNet.set_layer(4, [[784, "Input"], [200, "Sigmoid"], [200, "Sigmoid"], [10, "Softmax"]])
	myNet.set_learning_rate(0.1)
	myNet.set_momentum(0.5)
	myNet.set_l2_reg(0.1)
	myNet.set_dropout_rate(0.0)
	myNet.initNetwork()
	epochs = 250
	#
	[loss, closs, vloss, vcloss] = myNet.train_and_validate(X_train, y_train, X_test, y_test, epochs)
	#
	fig = plot_error(myNet, loss, vloss, "Cross-entropy Loss")
	fig.savefig('../Results/HW1/ce_two_layer_best.png', format='png')
	fig = plot_error(myNet, closs, vcloss, "Misclassification Error")
	fig.savefig('../Results/HW1/mc_two_layer_best.png', format='png')
	fig = myNet.visualize_layer(1)
	fig.savefig('../Results/HW1/visualize_weight_two_layer_best.png', format='png')

