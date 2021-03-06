#include <vector>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include "../cpu/QAPDataTypes.h"

__global__
void calculateQAPValues(unsigned dimension, unsigned permutationsCount, unsigned *weights, unsigned *distances,
                        unsigned *permutations, unsigned *values);

__global__
void performMovement(unsigned dimension, unsigned permutationsCount, unsigned distance, unsigned *permutations,
                     const unsigned *values, unsigned *nextPermutations,
                     unsigned *pmxBuffers, curandState *randomStates);

__global__
void copyPermutations(unsigned dimension, unsigned permutationsCount, unsigned *permutations, unsigned *nextPermutations);

__global__
void initializeRandomStates(curandState* devStates, unsigned statesCount, unsigned long seed);

__global__
void localSearch(unsigned dimension, unsigned permutationsCount, unsigned *permutations,
                 unsigned *nextPermutations, curandState *randomStates, unsigned *weights, unsigned *distances);

__global__
void permutationCheck(unsigned dimension, unsigned permutationsCount, unsigned *permutations, bool* result);

__global__
void findBestValueInEachBlock(unsigned valuesPerBlock, const unsigned *values, unsigned* output);

__global__
void findBestValue(unsigned valuesCount, const unsigned *values, unsigned* output);


unsigned* allocateData(const std::vector<std::vector<unsigned>> &data);
unsigned* allocateArray(unsigned size);







