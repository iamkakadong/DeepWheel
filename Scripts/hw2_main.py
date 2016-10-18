from Helper import Loader
from NetworkStructure import Network

if __name__ == '__main__':
	[X_train, y_train] = Loader.loadBinaryData("../data/digitstrain.txt")
	[X_val, y_val] = Loader.loadBinaryData("../data/digitsvalid.txt")
	[X_test, y_test] = Loader.loadBinaryData("../data/digitstest.txt")

	myNet = Network.Network()

	myNet.setLayer(3, [[784, "Input"], [100, "Sigmoid"], [784, "Sigmoid"]])
	myNet.setLearningRate(0.1)
	myNet.setMomentum(0.0)
	myNet.setL2Weight(0.0)
	myNet.setDropOutRate(0.0)
	myNet.initNetwork()
	epochs = 250

	[loss, closs, vloss, vcloss] = myNet.trainAndValidate(X_train, X_train, X_test, X_test, epochs)