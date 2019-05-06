#include <iostream>
#include <cuda_runtime.h>
#include "../src/gpu/PMX.h"


__global__ void kernel(unsigned dimension, unsigned* p1, unsigned* p2, unsigned start, unsigned end, unsigned* result) {
    pmx(dimension, p1, p2, start, end, result);
}

void test1() {
    unsigned p1[]{8, 4, 7, 3, 6, 2, 5, 1, 9, 0};
    unsigned p2[]{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

    unsigned * deviceP1;
    unsigned * deviceP2;
    unsigned * deviceResult;

    cudaMalloc(&deviceP1, sizeof(p1));
    cudaMalloc(&deviceP2, sizeof(p2));
    cudaMalloc(&deviceResult, sizeof(p2));

    cudaMemcpy(deviceP1, p1, sizeof(p1), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceP2, p2, sizeof(p2), cudaMemcpyHostToDevice);


    kernel<<<1, 1>>>(10, deviceP1, deviceP2, 3, 7, deviceResult);

    unsigned result[10];

    cudaMemcpy(p1, deviceP1, sizeof(p2), cudaMemcpyDeviceToHost);
    cudaMemcpy(p2, deviceP2, sizeof(p2), cudaMemcpyDeviceToHost);
    cudaMemcpy(result, deviceResult, sizeof(p2), cudaMemcpyDeviceToHost);

    for (auto val : p1)
        std::cout << val << " ";

    std::cout << std::endl;

    for (auto val : p2)
        std::cout << val << " ";

    std::cout << std::endl;

    for (auto val : result) // 0, 7, 4, 3, 6, 2, 5, 1, 8, 9
        std::cout << val << " ";

    std::cout << std::endl;

    cudaFree(deviceP1);
    cudaFree(deviceP2);
    cudaFree(deviceResult);
}

void test2() {
    unsigned p1[]{1, 2, 3, 4, 5, 6, 7, 8, 9};
    unsigned p2[]{4, 5, 2, 1, 8, 7, 6, 9, 3};

    unsigned * deviceP1;
    unsigned * deviceP2;
    unsigned * deviceResult;

    cudaMalloc(&deviceP1, sizeof(p1));
    cudaMalloc(&deviceP2, sizeof(p2));
    cudaMalloc(&deviceResult, sizeof(p2));

    cudaMemcpy(deviceP1, p1, sizeof(p1), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceP2, p2, sizeof(p2), cudaMemcpyHostToDevice);


    kernel<<<1, 1>>>(9, deviceP1, deviceP2, 3, 6, deviceResult);

    unsigned result[9];

    cudaMemcpy(p1, deviceP1, sizeof(p2), cudaMemcpyDeviceToHost);
    cudaMemcpy(p2, deviceP2, sizeof(p2), cudaMemcpyDeviceToHost);
    cudaMemcpy(result, deviceResult, sizeof(p2), cudaMemcpyDeviceToHost);

    for (auto val : p1)
        std::cout << val << " ";

    std::cout << std::endl;

    for (auto val : p2)
        std::cout << val << " ";

    std::cout << std::endl;

    for (auto val : result) // 1, 8, 2, 4, 5, 6, 7, 9, 3
        std::cout << val << " ";

    std::cout << std::endl;

    cudaFree(deviceP1);
    cudaFree(deviceP2);
    cudaFree(deviceResult);
}

void test3() {
    unsigned p1[]{1, 5, 2, 8, 7, 4, 3, 6};
    unsigned p2[]{4, 2, 5, 8, 1, 3, 6, 7};

    unsigned * deviceP1;
    unsigned * deviceP2;
    unsigned * deviceResult;

    cudaMalloc(&deviceP1, sizeof(p1));
    cudaMalloc(&deviceP2, sizeof(p2));
    cudaMalloc(&deviceResult, sizeof(p2));

    cudaMemcpy(deviceP1, p1, sizeof(p1), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceP2, p2, sizeof(p2), cudaMemcpyHostToDevice);


    kernel<<<1, 1>>>(8, deviceP1, deviceP2, 2, 4, deviceResult);

    unsigned result[8];

    cudaMemcpy(p1, deviceP1, sizeof(p2), cudaMemcpyDeviceToHost);
    cudaMemcpy(p2, deviceP2, sizeof(p2), cudaMemcpyDeviceToHost);
    cudaMemcpy(result, deviceResult, sizeof(p2), cudaMemcpyDeviceToHost);

    for (auto val : p1)
        std::cout << val << " ";

    std::cout << std::endl;

    for (auto val : p2)
        std::cout << val << " ";

    std::cout << std::endl;

    for (auto val : result) // 4, 5, 2, 8, 7, 3, 6, 1
        std::cout << val << " ";

    std::cout << std::endl;

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