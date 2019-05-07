#include "CUDAUtils.h"
#include <cuda_runtime.h>
#include <stdexcept>

void CUDAUtils::checkParameters(unsigned blocks, unsigned threads) {
    cudaDeviceProp deviceProperties{};
    auto result = cudaGetDeviceProperties(&deviceProperties, 0);

    if (result != cudaSuccess)
        throw std::runtime_error{"Could not get device properties"};

    if (blocks > deviceProperties.maxGridSize[0])
        throw std::runtime_error{"Number of blocks is higher than maximum "
                                    + std::to_string(deviceProperties.maxGridSize[0])};

    if (threads > deviceProperties.maxThreadsPerBlock)
        throw std::runtime_error{"Number of permutations per block is higher than maximum "
                                    + std::to_string(deviceProperties.maxThreadsPerBlock)};
}
