#ifndef DIANA_HAMMING_H
#define DIANA_HAMMING_H

#include <cuda_runtime.h>

__device__
unsigned hammingDistance(unsigned dimension, const unsigned* firstPermutation, const unsigned* secondPermutation){
    unsigned differences{0};

    for (unsigned i = 0; i < dimension; i++) {
        if (firstPermutation[i] != secondPermutation[i])
            differences++;
    }

    return differences;
}

#endif
