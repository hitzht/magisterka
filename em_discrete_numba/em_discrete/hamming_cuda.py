from numba import cuda
from typing import List


@cuda.jit('int64(int64[:], int64[:])', device=True)
def hamming_distance(first_permutation: List[int], second_permutation: List[int]):
    result = 0
    for i in range(len(first_permutation)):
        if first_permutation[i] != second_permutation[i]:
            result += 1

    return result
