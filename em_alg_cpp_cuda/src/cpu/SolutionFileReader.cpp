#include <fstream>
#include "SolutionFileReader.h"

SolutionFileReader::SolutionFileReader(const std::string &path) {
    std::ifstream inputFile{path};

    if (!inputFile.is_open())
        throw std::runtime_error("Could not open file: " + path);

    if (!(inputFile >> this->instanceSize))
        throw std::runtime_error("Could not read solution instance size");

    if (!(inputFile >> this->solutionValue))
        throw std::runtime_error("Could not read solution value");

    this->solution.clear();
    for (unsigned i = 0; i < this->instanceSize; i++) {
        unsigned value{0};
        if (!(inputFile >> value))
            throw std::runtime_error("Could not read value from solution file");

        this->solution.push_back(value - 1);
    }
}


unsigned SolutionFileReader::getInstanceSize() {
    return this->instanceSize;
}

unsigned SolutionFileReader::getSolutionValue() {
    return this->solutionValue;
}

Permutation<unsigned> SolutionFileReader::getSolution() {
    return this->solution;
}
