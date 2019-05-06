#include <iostream>
#include <cuda_runtime.h>

int main() {

    size_t free, total;


    cudaMemGetInfo(&free,&total);

    printf("%d KB free of total %d KB\n",free/1024,total/1024);

    unsigned* devicePointer{nullptr};


    unsigned blocks = 1000;
    unsigned permutations = 1000;
    unsigned dimension = 80;

    auto result = cudaMalloc(&devicePointer, blocks * permutations * dimension * 3 * sizeof(unsigned));

    if (result != cudaSuccess) {
        std::cerr << "error allocating memory" << std::endl;
    } else {
        std::cout << "ok" << std::endl;
    }

    cudaFree(devicePointer);

    return 0;
}
