#ifndef QAP_CPU_SOLUTIONFILEREADER_H
#define QAP_CPU_SOLUTIONFILEREADER_H

#include <string>
#include <fstream>
#include "QAPDataTypes.h"

class SolutionFileReader {
private:
    unsigned instanceSize;
    unsigned solutionValue;
    Permutation<unsigned> solution;

public:
    explicit SolutionFileReader(const std::string& path);

    unsigned getInstanceSize();
    unsigned getSolutionValue();
    Permutation<unsigned> getSolution();
};

#endif