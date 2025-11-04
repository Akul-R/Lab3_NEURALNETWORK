#pragma once
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <random>
#include <chrono>

class Matrix {
private:
	std::vector<std::vector<double>> data; //stores the data

public:
	//constructor classes
	Matrix() {
		//default constructor class, creates an empty matrix
		create(0, 0);
	}
	Matrix(std::vector<std::vector<double>> mat) {
		//creates matrix from a 2d vector input
		data = mat;
	}
	Matrix(int nrow, int ncol) {
		//create a nxm matrix
		create(nrow, ncol);
	}

	//other utility function
	void create(int nrow, int ncol) {
		data.assign(nrow, std::vector<double>(ncol, 0.0));
	}
	void randomise(double seed, double lower, double upper) {
		std::mt19937 generator(seed);
		std::uniform_real_distribution<> distribution(lower, upper);
		for (int row = 0; row < data.size(); row++) {
			for (int col = 0; col < data[0].size(); col++) {
				data[row][col] = distribution(generator);
			}
		}
	}
	void print() const {
		for (const auto& row : data) {
			for (double val : row) {
				std::cout << val << " ";
			}
			std::cout << "\n";
		}
	}

	//adding overloads for easy read and write access to data
	double& operator()(size_t row, size_t col) { return data[row][col]; }
	const double& operator()(size_t row, size_t col) const { return data[row][col]; } //read only version

	//methods to return the number of rows and columns in the matrix
	size_t rows() const { return data.size(); }
	size_t cols() const { return data[0].size(); }

	//matrix operations
	void transpose() {
		std::vector<std::vector<double>> result; 
		result.assign(cols(), std::vector<double>(rows(), 0.0)); 

		for (int rows = 0; rows < data.size(); rows++) {
			for (int cols = 0; cols < data[0].size(); cols++) {
				result[cols][rows] = data[rows][cols];
			}
		}

		data = result; //updates the data so now the matrix is transposed
	}
	void element_multiply(const Matrix& mat) {
		if (mat.cols() != cols() || mat.rows() != rows()) { //if the number of cols of mat1 doesnt match number of cols of mat2, matrix addition not possible
			throw::std::invalid_argument("MATRIX DIMENSIONS NOT COMPATABLE"); //throws error if they dont match, might change in future
		}

		std::vector<std::vector<double>> result;
		result.assign(rows(), std::vector<double>(cols(), 0.0));

		for (size_t row = 0; row < mat.rows(); row++) {
			for (size_t col = 0; col < mat.cols(); col++) {
				double sum = mat(row, col) * data[row][col];
				result[row][col] = sum;
			}
		}

		data = result; //updates the data so now stores result of element multiplication
	}

	//these operation will return a new Matrix. They are defined as a friend function
	friend Matrix operator*(const Matrix& mat1, const Matrix& mat2); //matrix multiplication
	friend Matrix operator*(const double scalar, const Matrix& mat); //matrix scalar multiplication
	friend Matrix operator+(const Matrix& mat1, const Matrix& mat2); //matrix addition
	friend Matrix operator-(const Matrix& mat1, const Matrix& mat2); //matrix subtraction

};

Matrix operator*(const Matrix& mat1, const Matrix& mat2) {
	if (mat1.cols() != mat2.rows()) { //if the number of rows of mat1 doesnt match number of columns of mat2, matrix multiplication not possible
		throw::std::invalid_argument("MATRIX DIMENSIONS NOT COMPATABLE"); //throws error if they dont match, might change in future
	}

	Matrix result; //the result of the multiplication, also a matrix 
	result.create(mat1.rows(), mat2.cols()); //resize the results matrix

	for (int row1 = 0; row1 < mat1.rows(); row1++) { //iterate through rows in mat1
		for (int col2 = 0; col2 < mat2.cols(); col2++) { //iterate through each column in mat2
			double sum = 0.0;
			for (int col1 = 0; col1 < mat1.cols(); col1++) { //iterate through each number in mat1
				double val1;
				double val2;

				val1 = mat1(row1, col1);
				val2 = mat2(col1, col2);

				sum += val1 * val2;
			}
			result(row1, col2) = sum;
		}
	}

	return result;
}

Matrix operator*(const double scalar, const Matrix& mat) {
	Matrix result; //the result of the multiplication, also a matrix 
	result.create(mat.rows(), mat.cols()); //resize the results matrix

	for (int row = 0; row < mat.rows(); row++) {
		for (int col = 0; col < mat.cols(); col++) {
			double sum = mat(row, col) * scalar;
			result(row, col) = sum;
		}
	}
	return result;
}

Matrix operator+(const Matrix& mat1, const Matrix& mat2) {
	if (mat1.cols() != mat2.cols() || mat1.rows() != mat2.rows()) { //if the number of cols of mat1 doesnt match number of cols of mat2, matrix addition not possible
		throw::std::invalid_argument("MATRIX DIMENSIONS NOT COMPATABLE"); //throws error if they dont match, might change in future
	}

	Matrix result; //the result of the multiplication, also a matrix 
	result.create(mat1.rows(), mat1.cols()); //resize the results matrix

	for (int row = 0; row < mat1.rows(); row++) {
		for (int col = 0; col < mat1.cols(); col++) {
			double sum = mat1(row, col) + mat2(row, col);
			result(row, col) = sum;
		}
	}

	return result;
}

Matrix operator-(const Matrix& mat1, const Matrix& mat2) {
	if (mat1.rows() != mat2.rows() || mat1.cols() != mat2.cols()) {
		throw::std::invalid_argument("MATRIX SUBTRACTION REQUIRES IDENTICAL DIMENSIONS");
	}

	Matrix result; //the result of the multiplication, also a matrix 
	result.create(mat1.rows(), mat1.cols()); //resize the results matrix

	for (size_t row = 0; row < mat1.rows(); row++) {
		for (size_t col = 0; col < mat1.cols(); col++) {
			double sum = mat1(row, col) - mat2(row, col);
			result(row, col) = sum;
		}
	}

	return result;
}
