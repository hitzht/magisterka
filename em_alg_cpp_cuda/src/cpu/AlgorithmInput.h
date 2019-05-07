#ifndef DIANA_ALGORITHMINPUT_H
#define DIANA_ALGORITHMINPUT_H

#include "QAPDataTypes.h"

struct AlgorithmInput {
    unsigned blocks;
    unsigned threads;
    unsigned dimension;
    unsigned iterations;
    unsigned neighborhoodDistance;
    Matrix<unsigned> weights;
    Matrix<unsigned> distances;
    std::vector<Permutation<unsigned>> permutations;
};

#endif
