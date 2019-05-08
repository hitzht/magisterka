#ifndef DIANA_QAP_GPU_H
#define DIANA_QAP_GPU_H

#include <cuda_runtime.h>

__device__
unsigned qapGPU(const unsigned dimension, const unsigned *weights, const unsigned *distances, const unsigned *permutation) {
    unsigned value{0};

    for (unsigned a = 0; a < dimension; a++) {
        for (unsigned b = 0; b < dimension; b++) {

            auto weight = weights[a * dimension + b];
            auto firstPoint = permutation[a];
            auto secondPoint = permutation[b];
            auto distance = distances[firstPoint * dimension + secondPoint];
            value += weight * distance;
        }
    }

    return value;
}

#endif //DIANA_QAP_GPU_H
