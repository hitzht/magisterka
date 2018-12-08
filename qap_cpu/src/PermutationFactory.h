#ifndef QAP_CPU_PERMUTATIONGENERATOR_H
#define QAP_CPU_PERMUTATIONGENERATOR_H

#include <random>
#include "QAPDataTypes.h"

class PermutationFactory {
private:
    std::random_device randomDevice;
    std::mt19937 generator;

public:
    PermutationFactory();

    Permutation<unsigned> get(unsigned size);
    std::vector<Permutation<unsigned>> get(unsigned size, unsigned count);
};


#endif
