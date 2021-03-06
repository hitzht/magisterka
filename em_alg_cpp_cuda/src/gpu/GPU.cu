#include "GPU.hpp"
#include <iostream>
#include <algorithm>
#include <time.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#include "QAP_GPU.h"
#include "Hamming.h"
#include "PMX.h"
#include "Repulsion.h"

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

__global__
void calculateQAPValues(unsigned dimension, unsigned permutationsCount, unsigned *weights, unsigned *distances,
                        unsigned *permutations, unsigned *values) {
    unsigned currentPermutationIndex = blockDim.x * blockIdx.x + threadIdx.x;

    if (currentPermutationIndex < permutationsCount) {
        auto currentPermutation = permutations + currentPermutationIndex * dimension;
        values[currentPermutationIndex] = qapGPU(dimension, weights, distances, currentPermutation);
    }
}

unsigned* allocateData(const std::vector<std::vector<unsigned>> &data) {
    unsigned* allocatedData{nullptr};
    auto dataDimension = data[0].size();

    auto result = cudaMalloc(&allocatedData, sizeof(unsigned) * data.size() * dataDimension);

    if (result != cudaSuccess)
        throw std::runtime_error{"Error while allocating memory on gpu, error code: " + std::to_string(result)};

    for(std::vector<std::vector<unsigned>>::size_type i = 0; i < data.size(); i++) {
        result = cudaMemcpy(allocatedData + i * dataDimension, data[i].data(), sizeof(unsigned) * dataDimension, cudaMemcpyHostToDevice);

        if (result != cudaSuccess)
            throw std::runtime_error{"Error while coping memory to gpu, error code: " + std::to_string(result)};
    }

    return allocatedData;
}

unsigned* allocateArray(unsigned size) {
    unsigned* deviceOutput{nullptr};
    auto result = cudaMalloc(&deviceOutput, sizeof(unsigned) * size);
    if (result != cudaSuccess)
        throw std::runtime_error{"Error while allocating memory for array on gpu, error code: " + std::to_string(result)};

    return deviceOutput;
}

__global__
void initializeRandomStates(curandState* states, unsigned statesCount, unsigned long seed) {
    unsigned id = blockDim.x * blockIdx.x + threadIdx.x;

    if (id < statesCount) {
        curand_init(seed, id, 0, states + id);
    }
}

__global__
void performMovement(unsigned dimension, unsigned permutationsCount, unsigned distance, unsigned *permutations,
                     const unsigned *values, unsigned *nextPermutations, unsigned *pmxBuffers, curandState *randomStates) {
    unsigned currentPermutationIndex = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned blockStart = blockDim.x * blockIdx.x;
    unsigned blockEnd = blockDim.x * (blockIdx.x + 1);
    
    if (currentPermutationIndex < permutationsCount) {
        auto currentPermutation = permutations + currentPermutationIndex * dimension;

        // find best permutation, thread 0 works, rest is waiting
        __shared__ unsigned bestPermutationIndex;

        if (threadIdx.x == 0) {
            bestPermutationIndex = 0;

            for (unsigned j = blockStart; j < blockEnd; j++) {
                if (values[j] < values[bestPermutationIndex]) {
                    bestPermutationIndex = j;
                }
            }
        }

        __syncthreads();

        // do not try to improve best permutation, just copy it to next array
        if (currentPermutationIndex == bestPermutationIndex) {
            auto nextPermutation = nextPermutations + currentPermutationIndex * dimension;

            for (unsigned i = 0; i < dimension; i++) {
                nextPermutation[i] = currentPermutation[i];
            }

            return;
        }


        auto pmxBuffer = pmxBuffers + currentPermutationIndex * dimension;

        // copy current permutation to next permutation
        auto nextPermutation = nextPermutations + currentPermutationIndex * dimension;

        for (unsigned j = 0; j < dimension; j++) {
            nextPermutation[j] = currentPermutation[j];
        }

        for (unsigned permutationIndex = blockStart; permutationIndex < blockEnd; permutationIndex++) {
            // current permutation can not improve itself
            if (permutationIndex == currentPermutationIndex)
                continue;

            auto sourcePermutation = permutations + permutationIndex * dimension;

            if (hammingDistance(dimension, currentPermutation, sourcePermutation) < distance) {
                if (values[permutationIndex] < values[currentPermutationIndex]) {
                    unsigned firstBound = (dimension - 1) * curand_uniform(randomStates + currentPermutationIndex);
                    unsigned secondBound = (dimension - 1) * curand_uniform(randomStates + currentPermutationIndex);
                    auto lower = MIN(firstBound, secondBound);
                    auto upper = MAX(firstBound, secondBound);

                    pmx(dimension, sourcePermutation, nextPermutation, lower, upper, pmxBuffer);

                    for (unsigned j = 0; j < dimension; j++) {
                        nextPermutation[j] = pmxBuffer[j];
                    }
                } else {
                    repulsion(dimension, sourcePermutation, nextPermutation, randomStates + currentPermutationIndex);
                }
            }
        }
    }
}

__global__
void localSearch(unsigned dimension, unsigned permutationsCount, unsigned *permutations,
                 unsigned *nextPermutations, curandState *randomStates, unsigned *weights, unsigned *distances) {
    unsigned currentPermutationIndex = blockDim.x * blockIdx.x + threadIdx.x;

    if (currentPermutationIndex < permutationsCount) {
        auto currentPermutation = permutations + currentPermutationIndex * dimension;
        auto nextPermutation = nextPermutations + currentPermutationIndex * dimension;

        for (unsigned i = 0; i < dimension; i++) {
            nextPermutation[i] = currentPermutation[i];
        }

        unsigned firstPosition = (dimension - 1) * curand_uniform(randomStates + currentPermutationIndex);
        unsigned secondPosition = (dimension - 1) * curand_uniform(randomStates + currentPermutationIndex);

        auto tmp = nextPermutation[firstPosition];
        nextPermutation[firstPosition] = nextPermutation[secondPosition];
        nextPermutation[secondPosition] = tmp;

        if (qapGPU(dimension, weights, distances, nextPermutation) < qapGPU(dimension, weights, distances, currentPermutation)) {
            for (unsigned i = 0; i < dimension; i++) {
                currentPermutation[i] = nextPermutation[i];
            }
        }
    }
}

__global__
void permutationCheck(unsigned dimension, unsigned permutationsCount, unsigned *permutations, bool* result) {
    unsigned currentPermutationIndex = blockDim.x * blockIdx.x + threadIdx.x;

    if (currentPermutationIndex < permutationsCount) {
        auto currentPermutation = permutations + currentPermutationIndex * dimension;

        for (unsigned i = 0; i < dimension; i++) {
            auto value = currentPermutation[i];

            for (unsigned j = i + 1; j < dimension; j++) {
                if (value == currentPermutation[j]) {
                    result[currentPermutationIndex] = false;
                    return;
                }
            }
        }

        result[currentPermutationIndex] = true;
    }
}

__global__
void copyPermutations(unsigned dimension, unsigned permutationsCount, unsigned *permutations, unsigned *nextPermutations) {
    unsigned currentPermutationIndex = blockDim.x * blockIdx.x + threadIdx.x;

    if (currentPermutationIndex < permutationsCount) {
        auto currentPermutation = permutations + currentPermutationIndex * dimension;
        auto currentnextPermutation = nextPermutations + currentPermutationIndex * dimension;

        for (unsigned i = 0; i < dimension; i++)
            currentPermutation[i] = currentnextPermutation[i];
    }
}

__global__
void findBestValueInEachBlock(unsigned valuesPerBlock, const unsigned *values, unsigned* output) {
    unsigned blockStart = valuesPerBlock * blockIdx.x;
    unsigned blockEnd = valuesPerBlock * (blockIdx.x + 1);
    auto bestValue = values[blockStart];

    for (unsigned i = blockStart; i < blockEnd; i ++) {
        if (values[i] < bestValue) {
            bestValue = values[i];
        }
    }

    output[blockIdx.x] = bestValue;
}

__global__
void findBestValue(unsigned valuesCount, const unsigned *values, unsigned* output) {
    auto bestValue = values[0];

    for (unsigned i = 0; i < valuesCount; i ++) {
        if (values[i] < bestValue) {
            bestValue = values[i];
        }
    }

    *output = bestValue;
}