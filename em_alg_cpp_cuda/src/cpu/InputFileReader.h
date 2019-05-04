#ifndef QAP_CPU_FILEREADER_H
#define QAP_CPU_FILEREADER_H

#include <string>
#include "QAPDataTypes.h"
#include <stdexcept>

class InputFileReader {
private:
    std::string path;
    unsigned instanceSize;
    Matrix<unsigned> weights;
    Matrix<unsigned> distances;

public:
    explicit InputFileReader(const std::string &path) :
        path(path) {
        if (this->path.empty())
            throw std::invalid_argument{"File path is empty"};

        this->readInput();
    }

    unsigned getInstanceSize() const;
    Matrix<unsigned> getWeights() const;
    Matrix<unsigned> getDistances() const;

private:
    void readInput();
    unsigned readInstanceSize(std::ifstream &input);
    Matrix<unsigned> readMatrix(std::ifstream &input, unsigned instanceSize);
    Matrix<unsigned> createMatrix(unsigned size);
};


#endif
