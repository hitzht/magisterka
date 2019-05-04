#ifndef DIANA_PMX_H
#define DIANA_PMX_H

#include <cuda_runtime.h>

__device__
unsigned findValueInPerumation(unsigned dimension, const unsigned* permutation, unsigned value) {
    for (unsigned i = 0; i < dimension; i++) {
        if (permutation[i] == value) {
            return i;
        }
    }

    return 0;
}

__device__
void pmx(unsigned dimension, unsigned* firstPermutation, unsigned* secondPermutation, unsigned start, unsigned end) {
    while(true) {

    }
}


#endif //DIANA_PMX_H
