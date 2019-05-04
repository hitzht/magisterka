#include <vector>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include "../cpu/QAPDataTypes.h"

std::vector<unsigned>
calculateOnGPU(unsigned dimension, unsigned iterations, unsigned distance, const Matrix<unsigned> &weights,
               const Matrix<unsigned> &distances, const std::vector<Permutation<unsigned>> &permutations);

unsigned* allocateData(const std::vector<std::vector<unsigned>> &data);
unsigned* allocateArray(unsigned size);

__global__
void initializeRandomStates(curandState* devStates, unsigned statesCount, unsigned long seed);

__global__
void emGPU(unsigned dimension, unsigned permutationsCount, unsigned distance, unsigned *weights, unsigned *distances,
           unsigned *permutations, unsigned *values, unsigned* temporaryBuffer, curandState* randomStates);

