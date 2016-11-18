from Helper import Loader
from Network import Network
from Layers.Sigmoid import Sigmoid
from Layers.Softmax import Softmax
from Layers.Noising import Noising
import matplotlib.pyplot as plt
import numpy as np
from Helper import LossFun
from scipy.io import loadmat

def plot_error2(loss_train, loss_valid, network):
	fig = plt.figure()
	plt.plot(loss_train, label='Training loss')
	plt.hold(True)
	plt.plot(loss_valid, label='Validation loss')
	plt.xlabel('Epoch Number')
	plt.ylabel('Cross-entropy')
	plt.title(network.print_struct())
	plt.legend()
	return fig

def plot_error(loss_train_pre, loss_train_naive, loss_valid_pre, loss_valid_naive, network):
	fig = plt.figure()
	plt.plot(loss_train_pre, label='Training loss of pre-trained network')
	plt.hold(True)
	plt.plot(loss_train_naive, label='Training loss of naive network')
	plt.plot(loss_valid_pre, label='Validation loss of pre-trained network')
	plt.plot(loss_valid_naive, label='Validation loss of naive network')
	plt.xlabel('Epoch Number')
	plt.ylabel('Misclassification rate')
	plt.title(network.print_struct())
	plt.legend()
	return fig

def rbm_exp():
	tmp = loadmat('../Results/HW2/rbm_weights.mat')
	weights = np.c_[np.array(tmp['model']['W'][0][0]).T, np.zeros([100, 1])]

	pretrain_network = Network.Network()
	naive_network = Network.Network()

	pretrain_layer = Sigmoid(784, 100, 0, 0, 0)
	pretrain_layer.weights = weights
	top_layer1 = Softmax(100, 10, 0, 0, 0)
	top_layer2 = Softmax(100, 10, 0, 0, 0)
	bot_layer = Sigmoid(784, 100, 0, 0, 0)
	pretrain_network.set_layer([pretrain_layer, top_layer1], [[784, "Input"], [100, "Sigmoid"], [10, "Softmax"]])
	naive_network.set_layer([bot_layer, top_layer2], [[784, "Input"], [100, "Sigmoid"], [10, "Softmax"]])
	pretrain_network.set_loss_func(['cross-entropy', 'misclassification'],
	                               [LossFun.cross_entropy, LossFun.misclassify_rate])
	naive_network.set_loss_func(['cross-entropy', 'misclassification'], [LossFun.cross_entropy, LossFun.misclassify_rate])
	pretrain_network.set_learning_rate(0.05)
	naive_network.set_learning_rate(0.05)
	epochs = 50

	[loss_train_pre, loss_valid_pre] = pretrain_network.train_and_validate(X_train, y_train, X_test, y_test, epochs)
	[loss_train_naive, loss_valid_naive] = naive_network.train_and_validate(X_train, y_train, X_test, y_test, epochs)
	fig = plot_error(loss_train_pre[1], loss_train_naive[1], loss_valid_pre[1], loss_valid_naive[1], pretrain_network)
	fig.savefig('../Results/HW2/rbm_init_accuracy.png', format='png')

def autoencoder_exp():
	my_net = Network.Network()

	n_hiddens = [50, 200, 500]
	for n_hidden in n_hiddens:
		l1 = Sigmoid(784, n_hidden, 0, 0, 0)
		l2 = Sigmoid(n_hidden, 784, 0, 0, 0)
		my_net.set_layer([l1, l2], [[784, "Input"], [n_hidden, "Sigmoid"], [784, "Sigmoid"]])
		my_net.set_learning_rate(0.01)
		my_net.set_loss_func(['Cross-entropy'], [LossFun.cross_entropy])
		epochs = 30

		[loss_train, loss_valid] = my_net.train_and_validate(X_train, X_train, X_test, X_test, epochs)
		fig = plot_error2(loss_train[0], loss_valid[0], my_net)
		fig.savefig('../Results/HW2/ae_train_' + str(n_hidden) + '.png', format='png')
		fig = my_net.visualize_layer(1)
		fig.savefig('../Results/HW2/visualize_weight_autoencoder_' + str(n_hidden) + '.png', format='png')

		my_net.forward_prop(X_train[450])
		output = my_net.output
		plt.matshow(np.reshape(output, [28, 28]), cmap=plt.cm.gray)
		plt.savefig('../Results/HW2/ae_recon_' + str(n_hidden) + '.png', format='png')

	# layer1 = Sigmoid(784, 100, 0, 0, 0)
	# layer2 = Sigmoid(100, 784, 0, 0, 0)
	# my_net.set_layer([layer1, layer2], [[784, "Input"], [100, "Sigmoid"], [784, "Sigmoid"]])
	# my_net.set_learning_rate(0.01)
	# my_net.set_loss_func(['cross-entropy'], [LossFun.cross_entropy])
	# my_net.set_weight_pair(0, 1)
	# epochs = 30
	#
	# [loss_train, loss_valid] = my_net.train_and_validate(X_train, X_train, X_test, X_test, epochs)

	# fig = plot_error2(loss_train[0], loss_valid[0], my_net)
	# fig.savefig('../Results/HW2/autoencoder_train.png', format='png')
	# fig = my_net.visualize_layer(1)
	# fig.savefig('../Results/HW2/visualize_weight_autoencoder.png', format='png')
	#
	# my_net.forward_prop(X_train[450])
	# output = my_net.output
	# plt.matshow(np.reshape(output, [28, 28]), cmap=plt.cm.gray)
	# plt.savefig('../Results/HW2/recon.png', format='png')

	# pretrain_network = Network.Network()
	# naive_network = Network.Network()
	#
	# top_layer1 = Softmax(100, 10, 0, 0, 0)
	# top_layer2 = Softmax(100, 10, 0, 0, 0)
	# bot_layer = Sigmoid(784, 100, 0, 0, 0)
	# pretrain_network.set_layer([layer1, top_layer1], [[784, "Input"], [100, "Sigmoid"], [10, "Softmax"]])
	# naive_network.set_layer([bot_layer, top_layer2], [[784, "Input"], [100, "Sigmoid"], [10, "Softmax"]])
	# pretrain_network.set_loss_func(['cross-entropy', 'misclassification'], [LossFun.cross_entropy, LossFun.misclassify_rate])
	# naive_network.set_loss_func(['cross-entropy', 'misclassification'], [LossFun.cross_entropy, LossFun.misclassify_rate])
	# pretrain_network.set_learning_rate(0.05)
	# naive_network.set_learning_rate(0.05)
	# epochs = 50
	#
	# [loss_train_pre, loss_valid_pre] = pretrain_network.train_and_validate(X_train, y_train, X_test, y_test, epochs)
	# [loss_train_naive, loss_valid_naive] = naive_network.train_and_validate(X_train, y_train, X_test, y_test, epochs)
	# fig = plot_error(loss_train_pre[1], loss_train_naive[1], loss_valid_pre[1], loss_valid_naive[1], pretrain_network)
	# fig.savefig('../Results/HW2/autoencoder_init_accuracy.png', format='png')

def denoising_autoencoder_exp():
	my_net = Network.Network()

	noise_layer = Noising(784, 0.5, True)
	n_hiddens = [50, 100, 200, 500]
	for n_hidden in n_hiddens:
		l1 = Sigmoid(784, n_hidden, 0, 0, 0)
		l2 = Sigmoid(n_hidden, 784, 0, 0, 0)
		my_net.set_layer([noise_layer, l1, l2], [[784, "Input"], [784, "Noising"], [n_hidden, "Sigmoid"], [784, "Sigmoid"]])
		my_net.set_learning_rate(0.01)
		my_net.set_loss_func(['Cross-entropy'], [LossFun.cross_entropy])
		my_net.set_weight_pair(1, 2)
		epochs = 30

		[loss_train, loss_valid] = my_net.train_and_validate(X_train, X_train, X_test, X_test, epochs)
		fig = plot_error2(loss_train[0], loss_valid[0], my_net)
		fig.savefig('../Results/HW2/denoising_train_' + str(n_hidden) + '.png', format='png')
		fig = my_net.visualize_layer(2)
		fig.savefig('../Results/HW2/visualize_weight_denoising_autoencoder_' + str(n_hidden) + '.png', format='png')

		my_net.forward_prop(X_train[450])
		output = my_net.output
		plt.matshow(np.reshape(output, [28, 28]), cmap=plt.cm.gray)
		plt.savefig('../Results/HW2/denoising_recon_' + str(n_hidden) + '.png', format='png')

	# l1 = Sigmoid(784, 100, 0, 0, 0)
	# l2 = Sigmoid(100, 784, 0, 0, 0)
	# my_net.set_layer([noise_layer, l1, l2], [[784, "Input"], [784, "Noising"], [100, "Sigmoid"], [784, "Sigmoid"]])
	# my_net.set_learning_rate(0.01)
	# my_net.set_loss_func(['Cross-entropy'], [LossFun.cross_entropy])
	# my_net.set_weight_pair(1, 2)
	# epochs = 30
	#
	# [loss_train, loss_valid] = my_net.train_and_validate(X_train, X_train, X_test, X_test, epochs)
	#
	# fig = plot_error2(loss_train[0], loss_valid[0], my_net)
	# fig.savefig('../Results/HW2/denoising_train.png', format='png')
	# fig = my_net.visualize_layer(2)
	# fig.savefig('../Results/HW2/visualize_weight_denoising_autoencoder_25.png', format='png')
	#
	# my_net.forward_prop(X_train[450])
	# output = my_net.output
	# plt.matshow(np.reshape(output, [28, 28]), cmap=plt.cm.gray)
	# plt.savefig('../Results/HW2/denoising_recon_25.png', format='png')

	# pretrain_network = Network.Network()
	# naive_network = Network.Network()
	#
	# top_layer1 = Softmax(100, 10, 0, 0, 0)
	# top_layer2 = Softmax(100, 10, 0, 0, 0)
	# bot_layer = Sigmoid(784, 100, 0, 0, 0)
	# pretrain_network.set_layer([l1, top_layer1], [[784, "Input"], [100, "Sigmoid"], [10, "Softmax"]])
	# naive_network.set_layer([bot_layer, top_layer2], [[784, "Input"], [100, "Sigmoid"], [10, "Softmax"]])
	# pretrain_network.set_loss_func(['cross-entropy', 'misclassification'],
	#                                [LossFun.cross_entropy, LossFun.misclassify_rate])
	# naive_network.set_loss_func(['cross-entropy', 'misclassification'], [LossFun.cross_entropy, LossFun.misclassify_rate])
	# pretrain_network.set_learning_rate(0.01)
	# naive_network.set_learning_rate(0.01)
	# epochs = 100
	#
	# [loss_train_pre, loss_valid_pre] = pretrain_network.train_and_validate(X_train, y_train, X_test, y_test, epochs)
	# [loss_train_naive, loss_valid_naive] = naive_network.train_and_validate(X_train, y_train, X_test, y_test, epochs)
	# fig = plot_error(loss_train_pre[1], loss_train_naive[1], loss_valid_pre[1], loss_valid_naive[1], pretrain_network)
	# fig.savefig('../Results/HW2/denoising_autoencoder_init_accuracy.png', format='png')

if __name__ == '__main__':
	[X_train, y_train] = Loader.load_binary_data("../data/digitstrain.txt")
	[X_test, y_test] = Loader.load_binary_data("../data/digitsvalid.txt")
	# [X_test, y_test] = Loader.load_binary_data("../data/digitstest.txt")

	# naive_exp()
	# rbm_exp()
	autoencoder_exp()
	denoising_autoencoder_exp()
