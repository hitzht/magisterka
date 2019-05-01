#include <vector>
#include <cuda_runtime.h>
#include "QAPDataTypes.h"

std::vector<unsigned> calculateOnGPU(unsigned dimension, const Matrix<unsigned>& weights, const Matrix<unsigned>& distances,
        const std::vector<Permutation<unsigned>>& permutations);

unsigned* allocData(const std::vector<std::vector<unsigned>>& data);

__global__
void qapGPU(unsigned dimension, unsigned permutationsCount, unsigned* weights, unsigned* distances,
            unsigned* permutations, unsigned* output);

__device__
unsigned qap(unsigned dimension, unsigned* weights, unsigned* distances, unsigned* permutation);