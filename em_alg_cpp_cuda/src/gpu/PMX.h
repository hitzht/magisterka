#ifndef DIANA_PMX_H
#define DIANA_PMX_H

#include <cuda_runtime.h>

#define UNKNOWN_VALUE 1000

__device__
unsigned findValueIndexInPerumation(unsigned dimension, const unsigned* permutation, unsigned value) {
    for (unsigned i = 0; i < dimension; i++) {
        if (permutation[i] == value) {
            return i;
        }
    }

    return 0;
}

__device__
bool isValueInRange(const unsigned* permutation,  unsigned start, unsigned end, unsigned value) {
    for (unsigned i = start; i <= end; i++) {
        if (permutation[i] == value) {
            return true;
        }
    }

    return false;
}

__device__
void switchValue(unsigned dimension, unsigned valueToSwitch, unsigned valueToWrite, unsigned* firstPermutation, unsigned* secondPermutation,
        unsigned start, unsigned end, unsigned* output) {
    while(true) {
        auto indexInSecondParent = findValueIndexInPerumation(dimension, secondPermutation, valueToSwitch);
        auto valueFromFistParent = firstPermutation[indexInSecondParent];
        indexInSecondParent= findValueIndexInPerumation(dimension, secondPermutation, valueFromFistParent);

        if ((start <= indexInSecondParent) && (indexInSecondParent <= end)) {
            valueToSwitch = valueFromFistParent;
        } else {
            output[indexInSecondParent] = valueToWrite;
            break;
        }
    }
}

__device__
void pmx(unsigned dimension, unsigned* firstPermutation, unsigned* secondPermutation, unsigned start, unsigned end,
         unsigned* output) {
    for (unsigned i = 0; i < dimension; i++) {
        output[i] = UNKNOWN_VALUE;
    }

    for (unsigned i = start; i <= end; i++) {
        output[i] = firstPermutation[i];
    }

    for (unsigned i = start; i <= end; i++) {
        auto value = secondPermutation[i];
        if (!isValueInRange(firstPermutation, start, end, value)) {
            switchValue(dimension, value, value, firstPermutation, secondPermutation, start, end, output);
        }
    }

    for (unsigned i = 0; i < dimension; i++) {
        if (output[i] == UNKNOWN_VALUE) {
            output[i] = secondPermutation[i];
        }
    }
}


#endif //DIANA_PMX_H
