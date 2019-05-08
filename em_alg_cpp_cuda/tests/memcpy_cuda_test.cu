#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

int main() {
    unsigned long size = 1000 * 100 * 20 * 3;

    std::vector<unsigned> v(size, 999);

    unsigned* devicePtr;

    auto res = cudaMalloc(&devicePtr, size * sizeof(unsigned));
    if (res != cudaSuccess) {
        std::cerr << "could not alloc memory" << std::endl;
        return 1;
    }

    auto now1 = std::chrono::system_clock::now();
    cudaMemcpy(devicePtr, v.data(), size * sizeof(unsigned), cudaMemcpyHostToDevice);
    auto now2 = std::chrono::system_clock::now();

    std::cout << "coping to device took " << std::chrono::duration_cast<std::chrono::milliseconds>(now2 - now1).count() << std::endl;

    now1 = std::chrono::system_clock::now();
    cudaMemcpy(v.data(), devicePtr, size * sizeof(unsigned), cudaMemcpyDeviceToHost);
    now2 = std::chrono::system_clock::now();

    std::cout << "coping to host took " << std::chrono::duration_cast<std::chrono::milliseconds>(now2 - now1).count() << std::endl;

    return 0;
}