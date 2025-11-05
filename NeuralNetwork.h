#pragma once
#include <iostream>
#include <iomanip>
#include <fstream>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
#include <bitset>
#include "Matrix.h"

const double MAX_EXP_ARG = 709.7827; // Prevents exp() overflow
const double MIN_EXP_ARG = -709.7827; // Prevents exp() overflow

namespace nn_utils {
	Matrix sigmoid(const Matrix& mat);
	Matrix dsigmoid(const Matrix& mat);
	Matrix relu(const Matrix& mat);
	Matrix drelu(const Matrix& mat);
	Matrix softmax(const Matrix& mat);
	Matrix glorot(int input, int output);
	Matrix he_init(int input, int output);
	int argmax(const Matrix& mat); //returns the index with highest probability
}

Matrix nn_utils::sigmoid(const Matrix& mat) {
	std::vector<std::vector<double>> result_data(mat.rows(), std::vector<double>(mat.cols()));
	for (int i = 0; i < mat.rows(); ++i)
	{
		for (int j = 0; j < mat.cols(); ++j)
		{
			result_data[i][j] = 1.0 / (1.0 + exp(-mat(i, j)));
		}
	}
	return Matrix(result_data);
}

Matrix nn_utils::dsigmoid(const Matrix& mat) {
	std::vector<std::vector<double>> result_data(mat.rows(), std::vector<double>(mat.cols()));
	Matrix sigmat = nn_utils::sigmoid(mat);
	for (int i = 0; i < mat.rows(); ++i)
	{
		for (int j = 0; j < mat.cols(); ++j)
		{
			result_data[i][j] = sigmat(i, j) * (1.0 - sigmat(i, j));
		}
	}
	return Matrix(result_data);
}

Matrix nn_utils::relu(const Matrix& mat) {
	std::vector<std::vector<double>> result_data;
	result_data.assign(mat.rows(), std::vector<double>(mat.cols(), 0.0));

	for (int r = 0; r < mat.rows(); ++r) {
		for (int c = 0; c < mat.cols(); ++c) {
			double val = mat(r, c);
			result_data[r][c] = (val > 0.0) ? val : 0.0;
		}
	}

	return Matrix(result_data);
}

Matrix nn_utils::drelu(const Matrix& mat) {
	std::vector<std::vector<double>> result_data;
	result_data.assign(mat.rows(), std::vector<double>(mat.cols(), 0.0));

	for (int r = 0; r < mat.rows(); ++r) {
		for (int c = 0; c < mat.cols(); ++c) {
			double val = mat(r, c);
			result_data[r][c] = (val > 0.0) ? 1.0 : 0.0;
		}
	}

	return Matrix(result_data);
}

Matrix nn_utils::softmax(const Matrix& mat) {
	std::vector<std::vector<double>> result_data(
		mat.rows(), std::vector<double>(mat.cols(), 0.0)
	);

	// Handle column-wise softmax (for column vectors)
	for (int c = 0; c < mat.cols(); ++c) {
		double max_value = mat(0, c);
		for (int r = 1; r < mat.rows(); ++r)
			if (mat(r, c) > max_value)
				max_value = mat(r, c);

		double sum_exp = 0.0;
		for (int r = 0; r < mat.rows(); ++r)
			sum_exp += std::exp(mat(r, c) - max_value);

		for (int r = 0; r < mat.rows(); ++r)
			result_data[r][c] = std::exp(mat(r, c) - max_value) / sum_exp;
	}

	return Matrix(result_data);
}

Matrix nn_utils::glorot(int input, int output) {
	double fan_in = input;
	double fan_out = output;

	double stdev = std::sqrt(2.0 / (fan_in + fan_out));
	std::random_device rd;
	std::mt19937 generator(rd());

	std::normal_distribution<double> distribution(0.0, stdev);

	Matrix result(output, input);
	for (int row = 0; row < result.rows(); ++row) {
		for (int col = 0; col < result.cols(); ++col) {
			result(row, col) = distribution(generator);
		}
	}
	return result;
}

Matrix nn_utils::he_init(int input, int output) {
	double fan_in = static_cast<double>(input);
	double stddev = std::sqrt(2.0 / fan_in); // He initialization formula

	std::random_device rd;
	std::mt19937 generator(rd());
	std::normal_distribution<double> distribution(0.0, stddev);

	Matrix result(output, input);
	for (int r = 0; r < result.rows(); ++r) {
		for (int c = 0; c < result.cols(); ++c) {
			result(r, c) = distribution(generator);
		}
	}

	return result;
}

int nn_utils::argmax(const Matrix& mat) {
	if (mat.cols() != 1) {
		std::cerr << "ARGMAX ONLY SUPPORTS MATRICES WITH 1 COLUMN";
	}

	int max_index = 0;
	double current_max = mat(0, 0); //takes the first number in the matrix and compares

	for (int rows = 0; rows < mat.rows(); ++rows) {
		if (mat(rows, 0) > current_max) {
			current_max = mat(rows, 0);
			max_index = rows;
		}
	}

	return max_index;
}

class Layer {
public:
	Matrix weights;
	Matrix bias;
	Layer(int input_size, int output_size, int init_func) {
		switch(init_func) {
		case(0):
			weights = nn_utils::he_init(input_size, output_size); //init func for ReLu
			bias = Matrix(output_size, 1); //making a bias matrix
			bias.randomise(2, 0, 0.2); //randomising with small positive numbers as ReLu favours this
			break;
		case(1):
			weights = nn_utils::glorot(input_size, output_size); //init func for sigmoid
			bias = Matrix(output_size, 1); //making a bias matrix with all 0 as sigmoid favour this
			break;
		default:
			//by default, choose relu because its better and im too lazy to put in a loop to ask user again
			weights = nn_utils::he_init(input_size, output_size); //init func for ReLu
			bias = Matrix(output_size, 1); //making a bias matrix
			bias.randomise(2, 0, 0.2); //randomising with small positive numbers as ReLu favours this
			break;
		}
	}
	Layer() {
		weights = Matrix();
		bias = Matrix();
	}
};


class ANN {
private:
	std::vector<int> layerSizes; //A vector describing the size of your neural network layers (eg, 4, 8, 16)
	std::vector<Layer> layers; //The layers in the network (well more like the connections between them tbh)

	std::vector<Matrix> activations; //Output of each layer (after non linear function applied)
	std::vector<Matrix> net_input; //Output of each layer (before non linear function applied)

	int nl_func = 0; //chooses the non linear function applied, 0 for ReLu and 1 for Sigmoid

	int nbits = 4; //number of bits in input layer, set to 4 by default

	Matrix feedforward(const Matrix& input) {
		//Z[L] = (W[L]*A[L-1]) + B[l] 
		//A[L] = sig(Z[L])
		activations.clear();
		net_input.clear();

		activations.emplace_back(input); //the first activation is technically the input given

		for (int nlayer = 0; nlayer < layers.size() - 1; ++nlayer) {
			//we apply sigmoid to all hidden layers
			Matrix prev_activation = activations.back();
			Matrix z = layers[nlayer].weights * prev_activation; // (W[L]*A[L-1])
			z = z + layers[nlayer].bias;

			net_input.emplace_back(z);

			Matrix layer_activation;
			switch (nl_func) {
			case(1):
				layer_activation = nn_utils::sigmoid(z); //apply non linear function to z
				break;
			default:
				//again, default is relu because its better and im lazy
				layer_activation = nn_utils::relu(z); //apply non linear function to z
				break;
			}

			activations.emplace_back(layer_activation);
		}

		Matrix final_activation = layers.back().weights * activations.back(); //we calculate the final activation from output layer
		final_activation = final_activation + layers.back().bias;
		net_input.emplace_back(final_activation);

		final_activation = nn_utils::softmax(final_activation); //apply non linear softmax function to the activation
		activations.emplace_back(final_activation);

		return final_activation;
	}

	double update(const Matrix& target, double training_rate) {
		/*
		output error = output activation - target
		hidden layer error[L] = ((T(W[L+1])*error[L+1])) em dsig(Z[L])

		*/

		std::vector<Matrix> layer_errors; //stored the error (delta) of each layer in a vector
		Matrix output_layer_error = activations.back() - target;

		layer_errors.emplace_back(output_layer_error);

		//calculating hidden layer errors now
		int total_layers = layers.size();
		for (int nlayer = total_layers - 2; nlayer >= 0; --nlayer) {
			Matrix prev_layer_weights = layers[nlayer + 1].weights;
			prev_layer_weights.transpose(); //transpose the weight matrix
			Matrix hidden_layer_error = prev_layer_weights * layer_errors.back();

			Matrix derivative_func;
			switch (nl_func) {
			case(1):
				derivative_func = nn_utils::dsigmoid(net_input[nlayer]);
				break;
			default:
				//again, same reason. if you couldnt tell, i love relu
				derivative_func = nn_utils::drelu(net_input[nlayer]);
				break;
			}
			hidden_layer_error.element_multiply(derivative_func);

			layer_errors.emplace_back(hidden_layer_error);
		}

		std::reverse(layer_errors.begin(), layer_errors.end()); //reverses the vector to make working with it easier

		std::vector<Matrix> weight_grad;
		std::vector<Matrix> bias_grad;

		//calculating the gradients (dw and db)
		for (int nlayers = layer_errors.size() - 1; nlayers >= 0; --nlayers) {
			//dw[L] = error[L]*(T(A[L-1]))
			//db[L] = error[L]

			Matrix dw = layer_errors[nlayers];
			Matrix prev_layer_activation = activations[nlayers];
			prev_layer_activation.transpose();
			dw = dw * prev_layer_activation;
			weight_grad.emplace_back(dw);

			Matrix db = layer_errors[nlayers];
			bias_grad.emplace_back(db);
		}

		std::reverse(weight_grad.begin(), weight_grad.end());
		std::reverse(bias_grad.begin(), bias_grad.end());

		//applying the weight/bias changes
		for (int nlayer = layers.size() - 1; nlayer >= 0; --nlayer) {
			Matrix scaled_weight = training_rate * weight_grad[nlayer];
			Matrix scaled_bias = training_rate * bias_grad[nlayer];

			Matrix new_weight = layers[nlayer].weights - scaled_weight;
			Matrix new_bias = layers[nlayer].bias - scaled_bias;

			layers[nlayer].weights = new_weight;
			layers[nlayer].bias = new_bias;
		}

		//calculate cce
		const double EPSILON = 1e-12;

		double cce_loss = 0.0; //cross entropy loss
		Matrix output_layer_activation = activations.back();

		for (int row = 0; row < target.rows(); ++row) {
			if (target(row, 0) == 1.0) { // finding the index where the 1 is in target

				double predicted_prob = output_layer_activation(row, 0);

				// Ensure the value is between epsilon and 1 - epsilon
				double clipped_prob = std::max(predicted_prob, EPSILON);
				clipped_prob = std::min(clipped_prob, 1.0 - EPSILON);
				cce_loss = -std::log(clipped_prob);

				return cce_loss;
			}
		}

		return cce_loss;
	}

	void init_layer(std::vector<int> sizes) {
		layers.clear(); //clears the layers first every time its called
		layerSizes = sizes;
		for (int nlayer = 0; nlayer < sizes.size() - 1; ++nlayer) {
			
			/*
			if sizes = (4, 8, 16)
			1st layer has 4 inputs and 8 outputs
			2nd layer has 8 inputs and 16 outputs
			*/

			int layer_inputs = sizes[nlayer];
			int layer_outputs = sizes[nlayer + 1];

			Layer new_layer = Layer(layer_inputs, layer_outputs, nl_func);
			layers.emplace_back(new_layer); //add the new layer object to the layer vector
		}
	}

public:


	ANN(std::vector<int> sizes, std::string func, int bits) {
		//constructor method for the ANN
		if (sizes.size() < 2) { //checks if the given sizes vector has an input and output layer (at least 2 layers needed)
			throw::std::invalid_argument("NEURAL NETWORK MUST HAVE AT LEAST 2 LAYERS");
			return;
		}
		if (bits != sizes[0]) { //checks if number of bits and neurons in input layer match
			throw::std::invalid_argument("NUMBER OF INPUT BITS AND INPUT NEURONS DONT MATCH");
			return;
		}
		if (pow(2, bits) != sizes.back()) { //checks if the output layer is correct size
			throw::std::invalid_argument("INVALID OUTPUT LAYER SIZE");
			return;
		}

		nbits = bits;
		if (func == "relu") {
			nl_func = 0;
		}
		else if (func == "sigmoid") {
			nl_func = 1;
		}
		else { //invalid function given
			throw::std::invalid_argument("UNKNOWN ACTIVATION FUNCTION");
			return;
		}

		init_layer(sizes);
		
	}

	void training_start(int max_epoch, int out_epoch, double training_rate) {
		int epoch = 0;
		const int max_bit = 64; //just putting this number as a max for the std::bitset, you would be insane to even try inputs close to this size
		const int bits = nbits; //fetching the number of input bits 

		std::mt19937 generator(2);
		std::uniform_int_distribution<> distribution(0, pow(2, bits) - 1); //adjusts max value depending on number of bits
		while (epoch < max_epoch) {
			int input = distribution(generator);
			Matrix input_mat(bits, 1); //creating input matrix
			Matrix target_mat(pow(2, bits), 1); //creating target matrix
			target_mat(input, 0) = 1;
			std::string input_binary = std::bitset<max_bit>(input).to_string(); //converts the input to binary, 2 -> 0010
			input_binary = input_binary.substr(max_bit - bits); //we do all this because bitset doesnt like const int bits
			for (int i = 0; i < input_binary.length(); ++i) {
				int num = input_binary[i] - '0';
				input_mat(i, 0) = num + 0.001; //mapping the binary string to the input matrix
			}

			Matrix output_mat = feedforward(input_mat);
			int output = nn_utils::argmax(output_mat);
			double error = update(target_mat, training_rate);

			if (epoch % out_epoch == 0) {
				std::cout << "Epoch: " << std::setw(5) << epoch << "| ";
				std::cout << "Input: " << std::setw(5) << input << "| ";
				std::cout << "Output: " << std::setw(5) << output << "| ";
				std::cout << "CCE: " << std::setw(6) << error << "\n";
			}
			epoch++;
		}

	}

	void testing(bool print_all_confidence) {
		const int bits = nbits;

		while (true) {
			std::string input;
			while (true) {
				std::cout << "\n\nEnter a binary number (enter e to exit): ";
				std::cin >> input;
				if (input == "e") {
					exit(0);
				}
				else if (input.size() != nbits) {
					std::cout << "Enter Valid Number!";
				}
				else {
					break;
				}
			}
			std::cout << "Input = " << std::stoi(input, nullptr, 2) << "\n";
			Matrix input_mat(bits, 1);
			for (int i = 0; i < input.length(); ++i) {
				int num = input[i] - '0';
				input_mat(i, 0) = num + 0.001; //mapping the binary string to the input matrix
			}
			Matrix outputmat = feedforward(input_mat);
			if (print_all_confidence) {
				std::cout << "\n-----------------------------\n'Condfidence' value for each number\n";
				for (int row = 0; row < outputmat.rows(); ++row) {
					std::cout << std::setw(2) << row << ": " << std::setw(4) << std::fixed << std::setprecision(2) << (outputmat(row, 0) * 100) << "%\n";
				}
				std::cout << "\n-----------------------------\n";
			}

			int output = nn_utils::argmax(outputmat);
			std::cout << "Program thinks " << input << " is the number " << output << " with "
				<< std::setw(4) << std::fixed << std::setprecision(2) << (outputmat(output, 0) * 100) << "% Confidence";
			std::cout << "\n-----------------------------\n";
		}
	}

	void save(const std::string& weight_file_name = "weights.wts", const std::string& bias_file_name = "bias.bss") const {
		//2 seperate files to save weights and biases. use headers to seperate the weights/biases per layer.
		std::ofstream wout(weight_file_name, std::ios::trunc); //creates the file, overwrites it if it exists
		std::ofstream bout(bias_file_name, std::ios::trunc); //wout for weight file, bout for bfile

		if (!wout.is_open() or !bout.is_open()) {
			std::cerr << "\nERROR, COULD NOT OPEN SPECIFIED FILES\n";
			return;
		}

		//write the layer weights and biases
		for (int l = 0; l < layers.size(); ++l) {
			wout << "#LAYER " << l << " WEIGHTS:\n";
			//writing weights
			for (int row = 0; row < layers[l].weights.rows(); ++row) {
				for (int col = 0; col < layers[l].weights.cols(); ++col) {
					wout << layers[l].weights(row, col) << " ";
				}
				wout << "\n";
			}
			wout << "\n";
			
			//writing biases
			bout << "#LAYER " << l << " BIASES:\n";
			for (int row = 0; row < layers[l].bias.rows(); ++row) {
				for (int col = 0; col < layers[l].bias.cols(); ++col) {
					bout << layers[l].bias(row, col) << " ";
				}
				bout << "\n";
			}
			bout << "\n";
		}
		wout << "\n" << "#END";
		bout << "\n" << "#END";

		std::cout << "\nSAVE WAS SUCCESSFUL\n";
		return;
	}

	void load(const std::string weight_file, const std::string bias_file) {
		std::ifstream win(weight_file); //win for weight input and bin for bias input
		std::ifstream bin(bias_file);

		if (!win.is_open() || !bin.is_open()) {
			std::cerr << "\nERROR, SPECIFIED FILE COULD NOT BE OPENED\n";
			return;
		}

		std::string line;
		std::vector<std::vector<double>> weight_data;
		std::vector<std::vector<double>> bias_data;

		std::vector<Matrix> weights;
		std::vector<Matrix> bias;


		//reading weight file
		while (std::getline(win, line)) {
			//# are header names, just kept to maintain readability of file
			if (line == "#END") { 
				if (!weight_data.empty()) {
					Matrix weight_mat(weight_data);
					weights.emplace_back(weight_mat);
					weight_data.clear();
				}
				break; 
			} //#end signifies the end of the file
			if (line.empty()) { continue; }
			if (!line.empty() and line[0] == '#') { //# signifies we move onto next layer
				if (!weight_data.empty()) {
					Matrix weight_mat(weight_data);
					weights.emplace_back(weight_mat);
					weight_data.clear();
				}
				continue; 
			}



			std::istringstream iss(line);
			std::vector<double> row_dat; //stores each row, we can then add this to the data
			double val;
			while (iss >> val) {
				row_dat.emplace_back(val);
			}
			if (!row_dat.empty()) {
				weight_data.emplace_back(row_dat);
			}

		}

		//reading bias file
		while (std::getline(bin, line)) {
			//# are header names, just kept to maintain readability of file
			if (line == "#END") {
				if (!bias_data.empty()) {
					Matrix bias_mat(bias_data);
					bias.emplace_back(bias_mat);
					bias_data.clear();
				}
				break;
			} //#end signifies the end of the file
			if (!line.empty() and line[0] == '#') { //# signifies we move onto next layer
				if (!bias_data.empty()) {
					Matrix bias_mat(bias_data);
					bias.emplace_back(bias_mat);
					bias_data.clear();
				}
				continue;
			}
			if (line.empty()) { continue; }


			std::istringstream iss(line);
			std::vector<double> row_dat; //stores each row, we can then add this to the data
			double val;
			while (iss >> val) {
				row_dat.emplace_back(val);
			}
			if (!row_dat.empty()) {
				bias_data.emplace_back(row_dat);
			}

		}

		//finding the size of the layers (eg {4, 8, 16}).
		std::vector<int> size;
		size.emplace_back(weights[0].cols());
		for (int i = 0; i < bias.size(); ++i) {
			int layer_size = bias[i].rows();
			size.emplace_back(layer_size);
		}

		init_layer(size);
		//apply loaded matrices to weight and bias matrices
		for (int l = 0; l < layers.size(); ++l) {
			layers[l].weights = weights[l];
			layers[l].bias = bias[l];
		}

		std::cout << "\nLOADED FROM FILE SUCCESSFULLY\n";

		win.close();
		bin.close();
	}

	void accuracy(bool print_all_wrong) {
		//function to test how accurate the model is across all inputs
		const int max_bit = 64;
		std::vector<std::vector<int>> incorrect; //stores the numbers the program gets incorrect as well as what the model guessed
		std::cout << "\nMEASURING ACCURACY OF MODEL\n";

		//iterate through each possible number and test if its correct
		for (int n = 0; n < pow(2, nbits); n++) {
			std::string input_binary = std::bitset<max_bit>(n).to_string(); //converts the input to binary, 2 -> 0010
			input_binary = input_binary.substr(max_bit - nbits); //we do all this because bitset doesnt like const int bits
			Matrix input_mat(nbits, 1);
			for (int i = 0; i < input_binary.length(); ++i) {
				int num = input_binary[i] - '0';
				input_mat(i, 0) = num + 0.001; //mapping the binary string to the input matrix
				//added a tiny offset as the network doesnt like it when activations are 0
				//0 = dead neurons (sometimes) :(
			}

			Matrix output = feedforward(input_mat);
			int output_num = nn_utils::argmax(output);
			if (output_num != n) {
				//ie the model guessed wrong
				std::vector<int> temp;
				temp.clear();
				//store the incorrect number and then what the program guessed
				temp.emplace_back(n);
				temp.emplace_back(output_num);
				incorrect.emplace_back(temp);
			}
		}

		if (print_all_wrong) {
			if (!incorrect.empty()) {
				std::cout << "\nNETWORK GOT THESE INPUTS INCORRECT:\n";
				std::cout << "----------------------------------------------\n\n";
				for (int i = 0; i < incorrect.size(); ++i) {
					std::cout << std::setw(4) << incorrect[i][0] << " (Model guessed " << incorrect[i][1] << ")\n";
				}
				std::cout << "\n----------------------------------------------\n\n";
			}
			else {
				std::cout << "\nNETWORK GOT NOTHING INCORRECT\n";
			}
		}

		int correct = pow(2, nbits) - incorrect.size();
		double accuracy_score = (correct / pow(2, nbits)) * 100; //calculate a percentage score for model

		std::cout << "\nMODEL IS " << std::fixed << std::setprecision(2) << accuracy_score << "% ACCURATE (" 
			<< correct << "/" << pow(2, nbits) << " Correct)\n";
	}
};
