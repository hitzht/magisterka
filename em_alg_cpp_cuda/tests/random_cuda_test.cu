#include <iostream>
#include <array>

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

__global__ void kernel(unsigned* data, curandState* randomState, unsigned long seed) {
    unsigned id = threadIdx.x;
    curand_init(seed, id, 0, randomState);

    for (unsigned i = 0; i < 10; i++) {
        data[i] = 99 * curand_uniform(randomState);
    }
}

int main() {
    unsigned numbers[10];

    unsigned* deviceNumbers;

    auto result = cudaMalloc(&deviceNumbers, sizeof(unsigned) * 10);
    if (result != cudaSuccess) {
        std::cerr << "could not alloc memory!" << std::endl;
        return 1;
    }

    curandState* randomState;
    result = cudaMalloc(&randomState, sizeof(curandState));
    if (result != cudaSuccess) {
        std::cerr << "could not alloc random state!" << std::endl;
        return 1;
    }

    unsigned long seed = time(nullptr);
    kernel<<<1, 1>>>(deviceNumbers, randomState, seed);

    cudaMemcpy(numbers, deviceNumbers, sizeof(unsigned) * 10, cudaMemcpyDeviceToHost);

    for (auto val : numbers) {
        std::cout << val << std::endl;
    }

    cudaFree(deviceNumbers);
    cudaFree(randomState);

    return 0;
}
