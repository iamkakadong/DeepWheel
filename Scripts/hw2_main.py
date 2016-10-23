from Helper import Loader
from NetworkStructure import Network
from Layers.Sigmoid import Sigmoid
from Layers.Softmax import Softmax
from Layers.Noising import Noising
import matplotlib.pyplot as plt
import numpy as np
from Helper import LossFun
from scipy.io import loadmat

def rbm_exp():
	tmp = loadmat('../RBM/rbm_weights.mat')
	weights = np.c_[np.array(tmp['model']['W'][0][0]).T, np.ones([100, 1])]

	my_net = Network.Network()

	layer1 = Sigmoid(784, 100, 0, 0, 0)
	layer1.weights = weights
	layer2 = Softmax(100, 10, 0, 0, 0)
	my_net.set_layer([layer1, layer2], [[784, "Input"], [100, "Sigmoid"], [10, "Softmax"]])
	my_net.set_learning_rate(0.01)
	my_net.set_loss_func(['cross-entropy', 'misclassification rate'], [LossFun.cross_entropy, LossFun.misclassify_rate])
	epochs = 10

	[loss_train, loss_valid] = my_net.train_and_validate(X_train, y_train, X_test, y_test, epochs)

def autoencoder_exp():
	my_net = Network.Network()

	layer1 = Sigmoid(784, 100, 0, 0, 0)
	layer2 = Sigmoid(100, 784, 0, 0, 0)
	my_net.set_layer([layer1, layer2], [[784, "Input"], [100, "Sigmoid"], [784, "Sigmoid"]])
	my_net.set_learning_rate(0.1)
	my_net.set_loss_func(['cross-entropy'], [LossFun.cross_entropy])
	epochs = 10

	[loss_train, loss_valid] = my_net.train_and_validate(X_train, X_train, X_test, X_test, epochs)

	fig = my_net.visualize_layer(1)
	fig.savefig('../Results/HW2/visualize_weight_autoencoder.png', format='png')

	my_net.forward_prop(X_train[450])
	output = my_net.output
	plt.matshow(np.reshape(output, [28, 28]), cmap=plt.cm.gray)
	plt.savefig('../Results/HW2/recon.png', format='png')

	toplayer = Softmax(100, 10, 0, 0, 0)
	my_net.set_layer([layer1, toplayer], [[784, "Input"], [100, "Sigmoid"], [10, "Softmax"]])
	my_net.set_loss_func(['cross-entropy', 'misclassification'], [LossFun.cross_entropy, LossFun.misclassify_rate])
	epochs = 10

	[loss_train, loss_valid] = my_net.train_and_validate(X_train, y_train, X_test, y_test, epochs)

def denoising_autoencoder_exp():
	my_net = Network.Network()

	noise_layer = Noising(784, 0.5, True)
	l1 = Sigmoid(784, 100, 0, 0, 0)
	l2 = Sigmoid(100, 784, 0, 0, 0)
	my_net.set_layer([noise_layer, l1, l2], [[784, "Input"], [784, "Noising"], [100, "Sigmoid"], [784, "Sigmoid"]])
	my_net.set_learning_rate(0.1)
	my_net.set_loss_func(['Cross-entropy'], [LossFun.cross_entropy])
	epochs = 10

	[loss_train, loss_valid] = my_net.train_and_validate(X_train, X_train, X_test, X_test, epochs)

	fig = my_net.visualize_layer(2)
	fig.savefig('../Results/HW2/visualize_weight_denoising_autoencoder.png', format='png')

	my_net.forward_prop(X_train[450])
	output = my_net.output
	plt.matshow(np.reshape(output, [28, 28]), cmap=plt.cm.gray)
	plt.savefig('../Results/HW2/denoising_recon.png', format='png')

	toplayer = Softmax(100, 10, 0, 0, 0)
	my_net.set_layer([l1, toplayer], [[784, "Input"], [100, "Sigmoid"], [10, "Softmax"]])
	my_net.set_loss_func(['cross-entropy', 'misclassification'], [LossFun.cross_entropy, LossFun.misclassify_rate])
	epochs = 10

	[loss_train, loss_valid] = my_net.train_and_validate(X_train, y_train, X_test, y_test, epochs)

if __name__ == '__main__':
	[X_train, y_train] = Loader.load_binary_data("../data/digitstrain.txt")
	[X_val, y_val] = Loader.load_binary_data("../data/digitsvalid.txt")
	[X_test, y_test] = Loader.load_binary_data("../data/digitstest.txt")

	# rbm_exp()
	autoencoder_exp()
	# denoising_autoencoder_exp()
