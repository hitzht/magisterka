#include "gpu.hpp"
#include <iostream>
#include <time.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#include "QAP_GPU.h"
#include "Hamming.h"
#include "PMX.h"

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

std::vector<unsigned>
calculateOnGPU(unsigned dimension, unsigned iterations, unsigned distance, const Matrix<unsigned> &weights,
               const Matrix<unsigned> &distances, const std::vector<Permutation<unsigned>> &permutations) {
    auto permutationsCount = permutations.size();

    std::vector<unsigned> calculatedValues{};
    calculatedValues.resize(permutationsCount);

    curandState* randomStates;
    auto result = cudaMalloc(&randomStates, sizeof(curandState) * permutationsCount);

    if (result != cudaSuccess)
        throw std::runtime_error("Error while allocating random states, error code: " + std::to_string(result));

    unsigned long seed = time(nullptr);
    initializeRandomStates<<<1, permutationsCount>>>(randomStates, permutationsCount, seed);

    auto deviceWeights = allocateData(weights);
    auto deviceDistances = allocateData(distances);
    auto devicePermutations = allocateData(permutations);
    auto deviceValues = allocateArray(permutations.size());
    auto deviceBuffer = allocateArray(permutations.size() * dimension);
    auto pmxBuffer = allocateArray(permutations.size() * dimension);

    for (unsigned iteration = 0; iteration < iterations; iteration++) {
        emGPU<<<1, permutationsCount>>>(dimension, permutationsCount, distance, deviceWeights, deviceDistances,
                devicePermutations, deviceValues, deviceBuffer, randomStates);
    }


    result = cudaMemcpy(calculatedValues.data(), deviceValues, sizeof(unsigned) * permutationsCount, cudaMemcpyDeviceToHost);
    if (result != cudaSuccess)
        throw std::runtime_error{"Error while coping output to host, error code: " + std::to_string(result)};

    cudaFree(deviceWeights);
    cudaFree(deviceDistances);
    cudaFree(devicePermutations);
    cudaFree(deviceValues);
    cudaFree(deviceBuffer);
    cudaFree(randomStates);


    return calculatedValues;
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
void emGPU(unsigned dimension, unsigned permutationsCount, unsigned distance, unsigned *weights, unsigned *distances,
           unsigned *permutations, unsigned *values, unsigned* temporaryBuffer, curandState* curandStates) {
    unsigned currentPermutationIndex = blockDim.x * blockIdx.x + threadIdx.x;

    if (currentPermutationIndex < permutationsCount) {
        auto permutation = permutations + currentPermutationIndex * dimension;
        values[currentPermutationIndex] = qapGPU(dimension, weights, distances, permutation);

        __syncthreads();

        // find best permutation, thread 0 works, rest is waiting
        __shared__ unsigned bestPermutationIndex;

        if (currentPermutationIndex == 0) {
            bestPermutationIndex = 0;

            for (unsigned j = 0; j < permutationsCount; j++) {
                if (values[j] < values[bestPermutationIndex]) {
                    bestPermutationIndex = j;
                }
            }
        }

        __syncthreads();

        // do not try to improve best permutation
        if (currentPermutationIndex == bestPermutationIndex)
            return;

        auto currentPermutation = permutations + currentPermutationIndex * dimension;

        // copy current permutation to temporary buffer
        auto buffer = temporaryBuffer + currentPermutationIndex * dimension;

        for (unsigned j = 0; j < dimension; j++) {
            buffer[j] = currentPermutation[j];
        }

        for (unsigned permutationIndex = 0; permutationIndex < permutationsCount; permutationIndex++) {
            // current permutation can not improve itself
            if (permutationIndex == currentPermutationIndex)
                continue;

            auto sourcePermutation = permutations + permutationIndex * dimension;

            if (hammingDistance(dimension, currentPermutation, sourcePermutation) < distance) {
                if (values[permutationIndex] < values[currentPermutationIndex]) {
                    unsigned firstBound = float(dimension - 1) * curand_uniform(curandStates + currentPermutationIndex);
                    unsigned secondBound = float(dimension - 1) * curand_uniform(curandStates + currentPermutationIndex);
                    auto lower = MIN(firstBound, secondBound);
                    auto upper = MAX(firstBound, secondBound);

                    pmx(dimension, sourcePermutation, buffer, lower, upper);
                } else {
                    // repulsion
                }
            }
        }

        __syncthreads();

        // replace current permutation with improved one
        for (unsigned j = 0; j < dimension; j++) {
            currentPermutation[j] = buffer[j];
        }
    }
}

