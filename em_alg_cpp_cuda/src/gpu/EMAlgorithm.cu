#include "EMAlgorithm.h"
#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#include "GPU.hpp"

unsigned EMAlgorithm::solve(const AlgorithmInput &input) {
    auto permutationsCount = input.permutations.size();
    std::vector<unsigned> calculatedValues(permutationsCount);

    curandState* randomStates;
    auto result = cudaMalloc(&randomStates, sizeof(curandState) * permutationsCount);

    if (result != cudaSuccess)
        throw std::runtime_error("Error while allocating random states, error code: " + std::to_string(result));

    unsigned long seed = time(nullptr);

    initializeRandomStates<<<input.blocks, input.threads>>>(randomStates, permutationsCount, seed);

    auto deviceWeights = allocateData(input.weights);
    auto deviceDistances = allocateData(input.distances);
    auto devicePermutations = allocateData(input.permutations);
    auto deviceValues = allocateArray(permutationsCount);
    auto deviceNextPermutations = allocateArray(permutationsCount * input.dimension);
    auto pmxBuffer = allocateArray(permutationsCount * input.dimension);

    unsigned bestPermutationValue{0};

    for (unsigned iteration = 0; iteration < input.iterations; iteration++) {
        std::cout << "iter " << iteration << std::endl;

        calculateQAPValues<<<input.blocks, input.threads>>>(input.dimension, permutationsCount, deviceWeights,
                                                           deviceDistances, devicePermutations, deviceValues);

        performMovement<<<input.blocks, input.threads>>>(input.dimension, permutationsCount, input.neighborhoodDistance,
                                                         devicePermutations, deviceValues, deviceNextPermutations, pmxBuffer, randomStates);

        copyPermutations<<<input.blocks, input.threads>>>(input.dimension, permutationsCount, devicePermutations, deviceNextPermutations);
    }

    result = cudaMemcpy(calculatedValues.data(), deviceValues, sizeof(unsigned) * permutationsCount, cudaMemcpyDeviceToHost);
    if (result != cudaSuccess)
        throw std::runtime_error{"Error while coping output to host, error code: " + std::to_string(result)};

    bestPermutationValue = *std::min_element(calculatedValues.begin(), calculatedValues.end());

    cudaFree(deviceWeights);
    cudaFree(deviceDistances);
    cudaFree(devicePermutations);
    cudaFree(deviceValues);
    cudaFree(deviceNextPermutations);
    cudaFree(randomStates);
    cudaFree(pmxBuffer);

    return bestPermutationValue;
}
