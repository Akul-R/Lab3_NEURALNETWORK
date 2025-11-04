#include "Matrix.h"
#include "NeuralNetwork.h"
#include <iostream>
#include <iomanip>
#include <string>
#include <random>
#include <bitset>

int main() {
	int max_epoch = 5000;
	double training_rate = 0.01; //i hate this f*cking variable so much (i set it to 0 by accident and spent 3+ hours wondering why my network was getting dumber with each epoch)
	std::mt19937 generator(2);
	ANN NN1({ 4,8,16 });
	int epoch = 0;

	//training loop
	std::uniform_int_distribution<> distribution(0, 15); //for 4 bit binary
	while (epoch < max_epoch) {
		std::cout << "Epoch: " << std::setw(5) << epoch << "| ";

		int input = distribution(generator);
		std::cout << "Input: " << std::setw(2) << input << "| ";

		Matrix input_mat(4, 1); //creating input matrix
		Matrix target_mat(16, 1); //creating target matrix
		target_mat(input, 0) = 1;
		std::string input_binary = std::bitset<4>(input).to_string(); //converts the input to binary, 2 -> 0010
		for (int i = 0; i < input_binary.length(); ++i) {
			int num = input_binary[i] - '0';
			input_mat(i, 0) = num+0.001; //mapping the binary string to the input matrix
		}

		Matrix output_mat = NN1.feedforward(input_mat);
		int output = nn_utils::argmax(output_mat);
		std::cout << "Output: " << std::setw(2) << output << "| ";

		double error = NN1.update(target_mat, training_rate);
		std::cout << "CCE: " << std::setw(6) << error << "\n";

		epoch++;
	}

	while (true) {
		std::cout << "\n\nEnter a binary number: ";
		std::string input;
		std::cin >> input;
		std::cout << "\n-----------------------------\n";
		Matrix input_mat(4, 1);
		for (int i = 0; i < input.length(); ++i) {
			int num = input[i] - '0';
			input_mat(i, 0) = num+0.001; //mapping the binary string to the input matrix
		}
		Matrix outputmat = NN1.feedforward(input_mat);
		outputmat.print();
		std::cout << "\n-----------------------------\n";
		int output = nn_utils::argmax(outputmat);
		std::cout << "Program thinks " << input << " is the number " << output << "\n\n";
	}
}