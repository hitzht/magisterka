#include <iostream>

#include <cuda_runtime.h>

#include "../src/gpu/Hamming.h"
#include "../../../../../../usr/local/cuda-10.1/targets/x86_64-linux/include/driver_types.h"

__global__ void kernel(unsigned dimension, unsigned* p1, unsigned* p2, unsigned* result) {
    *result = hammingDistance(dimension, p1, p2);
}

void test1() {
    unsigned p1[]{8, 4, 7, 3, 6, 2, 5, 1, 9, 0};
    unsigned p2[]{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

    unsigned* deviceP1;
    unsigned* deviceP2;
    unsigned* deviceResult;

    cudaMalloc(&deviceP1, sizeof(p1));
    cudaMalloc(&deviceP2, sizeof(p2));
    cudaMalloc(&deviceResult, sizeof(unsigned));

    cudaMemcpy(deviceP1, p1, sizeof(p1), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceP2, p2, sizeof(p2), cudaMemcpyHostToDevice);

    kernel<<<1, 1>>>(10, deviceP1, deviceP2, deviceResult);

    unsigned result;
    cudaMemcpy(&result, deviceResult, sizeof(unsigned), cudaMemcpyDeviceToHost);

    std::cout << std::boolalpha << " " << (result == 9) << std::endl;

    cudaFree(deviceP1);
    cudaFree(deviceP2);
    cudaFree(deviceResult);
}


void test2() {
    unsigned p1[]{1, 2, 3, 4, 5, 6, 7, 8, 9};
    unsigned p2[]{4, 5, 2, 1, 8, 7, 6, 3, 9};

    unsigned* deviceP1;
    unsigned* deviceP2;
    unsigned* deviceResult;

    cudaMalloc(&deviceP1, sizeof(p1));
    cudaMalloc(&deviceP2, sizeof(p2));
    cudaMalloc(&deviceResult, sizeof(unsigned));

    cudaMemcpy(deviceP1, p1, sizeof(p1), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceP2, p2, sizeof(p2), cudaMemcpyHostToDevice);

    kernel<<<1, 1>>>(9, deviceP1, deviceP2, deviceResult);

    unsigned result;
    cudaMemcpy(&result, deviceResult, sizeof(unsigned), cudaMemcpyDeviceToHost);

    std::cout << std::boolalpha << " " << (result == 8) << std::endl;

    cudaFree(deviceP1);
    cudaFree(deviceP2);
    cudaFree(deviceResult);
}

void test3() {
    unsigned p1[]{1, 2, 3, 4, 5};
    unsigned p2[]{1, 2, 3, 4, 5};

    unsigned* deviceP1;
    unsigned* deviceP2;
    unsigned* deviceResult;

    cudaMalloc(&deviceP1, sizeof(p1));
    cudaMalloc(&deviceP2, sizeof(p2));
    cudaMalloc(&deviceResult, sizeof(unsigned));

    cudaMemcpy(deviceP1, p1, sizeof(p1), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceP2, p2, sizeof(p2), cudaMemcpyHostToDevice);

    kernel<<<1, 1>>>(5, deviceP1, deviceP2, deviceResult);

    unsigned result;
    cudaMemcpy(&result, deviceResult, sizeof(unsigned), cudaMemcpyDeviceToHost);

    std::cout << std::boolalpha << " " << (result == 0) << std::endl;

    cudaFree(deviceP1);
    cudaFree(deviceP2);
    cudaFree(deviceResult);
}

int main() {
    test1();
    test2();
    test3();

    return 0;
}
