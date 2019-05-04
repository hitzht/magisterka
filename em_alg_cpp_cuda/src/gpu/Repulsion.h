#ifndef DIANA_REPULSION_H
#define DIANA_REPULSION_H

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "Hamming.h"

__device__
void repulsion(unsigned dimension, unsigned* firstPermutation, unsigned* secondPermutation, curandState* randomState) {
    auto distance = hammingDistance(dimension, firstPermutation, secondPermutation);
    unsigned newDistance = 0;
    unsigned iterations = 0;

    while (distance >= newDistance) {
        iterations++;

        if (iterations == dimension) {
            break;
        }

        unsigned firstIndex = float(dimension - 1) * curand_uniform(randomState);
        unsigned secondIndex = float(dimension - 1) * curand_uniform(randomState);

        auto tmp = secondPermutation[firstIndex];
        secondPermutation[firstIndex] = secondPermutation[secondIndex];
        secondPermutation[secondIndex] = tmp;

        newDistance = hammingDistance(dimension, firstPermutation, secondPermutation);
    }
}

#endif
