#include "NeuralNetwork.h"

int main() {
	int max_epoch = 5000;
	int out_epoch = 500; //how many epochs between prints

	const int bits = 4; //specify how many bit input you are doing. bear in mind, larger bits take a lot longer to train properly

	double training_rate = 0.01; //i hate this f*cking variable so much (i set it to 0 by accident and spent 3+ hours wondering why my network was getting dumber with each epoch)
	ANN NN1({4, 8, 16}, "relu", bits); //first arg is vector for layer and their sizes, 2nd is activation function and 3rd is number of bits
	int epoch = 0;

	NN1.training_start(max_epoch, out_epoch, training_rate);
	NN1.save("4BIT_weight.wts", "4BIT_bias.bss");
	NN1.accuracy(true);
	NN1.testing(true);
}


