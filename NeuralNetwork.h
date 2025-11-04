#pragma once
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
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
	std::vector<std::vector<double>> result_data;
	result_data.assign(mat.rows(), std::vector<double>(mat.cols(), 0.0));
	for (int rows = 0; rows < mat.rows(); ++rows)
	{
		for (int cols = 0; cols < mat.cols(); ++cols)
		{
			double z = 0.0;
			//clamping to prevent the output from being a very big or very small number
			if (mat(rows, cols) > MAX_EXP_ARG) {
				z = MAX_EXP_ARG;
			}
			else if (mat(rows, cols) < MIN_EXP_ARG) {
				z = MIN_EXP_ARG;
			}
			else {
				z = mat(rows, cols);
			}

			result_data[rows][cols] = 1.0 / (1.0 + exp(-z));
		}
	}
	Matrix result(result_data);
	return result;
}

Matrix nn_utils::dsigmoid(const Matrix& mat) {
	//derivative of sigmoid function is simply = sig(x)(1-sig(x))
	Matrix A = nn_utils::sigmoid(mat);

	std::vector<std::vector<double>> result_data;
	result_data.assign(mat.rows(), std::vector<double>(mat.cols(), 0.0));

	for (int rows = 0; rows < mat.rows(); ++rows)
	{
		for (int cols = 0; cols < mat.cols(); ++cols)
		{
			double a = A(rows, cols);
			result_data[rows][cols] = a * (1.0 - a);
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
	Layer(int input_size, int output_size) {
		weights = nn_utils::he_init(input_size, output_size);
		bias = Matrix(output_size, 1); //making a bias matrix with all 0
		bias.randomise(2, 0, 0.2);
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

public:

	ANN(std::vector<int> sizes) {
		//constructor method for the ANN
		if (sizes.size() < 2) { //checks if the given sizes vector has an input and output layer (at least 2 layers needed)
			throw::std::invalid_argument("NEURAL NETWORK MUST HAVE AT LEAST 2 LAYERS");
			return;
		}

		for (int nlayer = 0; nlayer < sizes.size() - 1; ++nlayer) {
			/*
			if sizes = (4, 8, 16)
			1st layer has 4 inputs and 8 outputs
			2nd layer has 8 inputs and 16 outputs
			*/

			int layer_inputs = sizes[nlayer];
			int layer_outputs = sizes[nlayer + 1];

			Layer new_layer = Layer(layer_inputs, layer_outputs);
			layers.emplace_back(new_layer); //add the new layer object to the layer vector
		}
	}

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

			Matrix layer_activation = nn_utils::relu(z); //apply non linear function to z
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
			Matrix prev_layer_weights = layers[nlayer+1].weights;
			prev_layer_weights.transpose(); //transpose the weight matrix
			Matrix hidden_layer_error =  prev_layer_weights * layer_errors.back();
			
			Matrix dsig = nn_utils::drelu(net_input[nlayer]);
			hidden_layer_error.element_multiply(dsig);

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
};