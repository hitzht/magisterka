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
    
    for (unsigned iteration = 0; iteration < input.iterations; iteration++) {
        // std::cout << "iter " << iteration << std::endl;

        calculateQAPValues<<<input.blocks, input.threads>>>(input.dimension, permutationsCount, deviceWeights,
                                                           deviceDistances, devicePermutations, deviceValues);

        performMovement<<<input.blocks, input.threads>>>(input.dimension, permutationsCount, input.neighborhoodDistance,
                                                         devicePermutations, deviceValues, deviceNextPermutations, pmxBuffer, randomStates);

        copyPermutations<<<input.blocks, input.threads>>>(input.dimension, permutationsCount, devicePermutations, deviceNextPermutations);

        localSearch <<<input.blocks, input.threads>>>(input.dimension, permutationsCount, devicePermutations, deviceNextPermutations, randomStates,
                deviceWeights, deviceDistances);
    }

    calculateQAPValues<<<input.blocks, input.threads>>>(input.dimension, permutationsCount, deviceWeights,
            deviceDistances, devicePermutations, deviceValues);

    unsigned* deviceResult{nullptr};
    cudaMalloc(&deviceResult, input.blocks * sizeof(unsigned));
    findBestValueInEachBlock<<<input.blocks, 1>>>(input.threads, deviceValues, deviceResult);

    findBestValue<<<1, 1>>>(input.blocks, deviceResult, deviceResult);

    unsigned bestPermutationValue{0};
    result = cudaMemcpy(&bestPermutationValue, deviceResult, sizeof(unsigned), cudaMemcpyDeviceToHost);
    if (result != cudaSuccess) {
        std::cerr << "Error while coping result to host, error code: " + std::to_string(result) << std::endl;
    }

    cudaFree(deviceWeights);
    cudaFree(deviceDistances);
    cudaFree(devicePermutations);
    cudaFree(deviceValues);
    cudaFree(deviceNextPermutations);
    cudaFree(randomStates);
    cudaFree(pmxBuffer);
    cudaFree(deviceResult);

    return bestPermutationValue;
}
