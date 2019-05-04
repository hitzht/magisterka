#include "InputFileReader.h"
#include <exception>
#include <fstream>
#include <iostream>

void InputFileReader::readInput() {
    std::ifstream input{this->path};

    if (!input.is_open())
        throw std::runtime_error{"Could not open file " + this->path};

    this->instanceSize = this->readInstanceSize(input);
    this->weights = this->readMatrix(input, this->instanceSize);
    this->distances = this->readMatrix(input, this->instanceSize);
}

unsigned InputFileReader::readInstanceSize(std::ifstream &input) {
    unsigned value{0};
    if (!(input >> value))
        throw std::runtime_error{"Could not read instance size"};

    return value;
}

Matrix<unsigned> InputFileReader::readMatrix(std::ifstream &input, const unsigned instanceSize) {
    auto matrix = this->createMatrix(instanceSize);

    for (unsigned i = 0; i < instanceSize; i++) {
        for (unsigned j = 0; j < instanceSize; j++) {
            unsigned value{0};
            if (!(input >> value))
                throw std::runtime_error{"Error while reading value from file " + this->path};

            matrix[i][j] = value;
        }
    }

    return matrix;
}

Matrix<unsigned> InputFileReader::createMatrix(const unsigned size) {
    if (size <= 0)
        throw std::runtime_error{"Invalid matrix size"};

    Matrix<unsigned> matrix;
    for (unsigned i = 0; i < size; i++)
        matrix.push_back(std::vector<unsigned>(size));

    return matrix;
}

unsigned int InputFileReader::getInstanceSize() const {
    return this->instanceSize;
}

Matrix<unsigned> InputFileReader::getWeights() const {
    return this->weights;
}

Matrix<unsigned> InputFileReader::getDistances() const {
    return this->distances;
}