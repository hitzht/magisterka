#include "gpu.hpp"
#include <iostream>
#include <device_launch_parameters.h>

std::vector<unsigned> calculateOnGPU(unsigned dimension, const Matrix<unsigned>& weights, const Matrix<unsigned>& distances,
                    const std::vector<Permutation<unsigned>>& permutations) {
    auto permutationsCount = permutations.size();

    std::vector<unsigned> calculatedValues{};
    calculatedValues.resize(permutationsCount);

    auto deviceWeights = allocData(weights);
    auto deviceDistances = allocData(distances);
    auto devicePermutations = allocData(permutations);

    unsigned* deviceOutput{nullptr};
    auto result = cudaMalloc(&deviceOutput, sizeof(unsigned) * permutationsCount);
    if (result != cudaSuccess)
        throw std::runtime_error{"Error while allocating memory for output on gpu, error code: " + std::to_string(result)};

    qapGPU<<<1, permutationsCount>>>(dimension, permutationsCount, deviceWeights, deviceDistances, devicePermutations, deviceOutput);

    result = cudaMemcpy(calculatedValues.data(), deviceOutput, sizeof(unsigned) * permutationsCount, cudaMemcpyDeviceToHost);
    if (result != cudaSuccess)
        throw std::runtime_error{"Error while coping output to host, error code: " + std::to_string(result)};

    cudaFree(deviceWeights);
    cudaFree(deviceDistances);
    cudaFree(devicePermutations);
    cudaFree(deviceOutput);

    return calculatedValues;
}

unsigned* allocData(const std::vector<std::vector<unsigned>>& data) {
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

__global__
void qapGPU(unsigned dimension, unsigned permutationsCount, unsigned* weights, unsigned* distances,
            unsigned* permutations, unsigned* output) {
    unsigned i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < permutationsCount) {
        auto permutation = permutations + i * dimension;
        output[i] = qap(dimension, weights, distances, permutation);
    }
}

__device__
unsigned qap(unsigned dimension, unsigned* weights, unsigned* distances, unsigned* permutation) {
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