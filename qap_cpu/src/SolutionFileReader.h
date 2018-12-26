#ifndef QAP_CPU_SOLUTIONFILEREADER_H
#define QAP_CPU_SOLUTIONFILEREADER_H

#include <string>
#include <fstream>
#include "QAPDataTypes.h"

class SolutionFileReader {
public:
    explicit SolutionFileReader(const std::string& path);

    unsigned getSolutionSize();
    unsigned getSolutionValue();
    Permutation<unsigned> getSolution();

};

#endif