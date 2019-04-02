from typing import List
from numba import cuda
from em_discrete.hamming_cuda import hamming_distance
from numba.cuda.random import xoroshiro128p_uniform_float64


@cuda.jit(None, device=True)
def repulsion(first_permutation: List[int], second_permutation: List[int], random_states):
    dimension = len(first_permutation)
    thread_id = cuda.threadIdx.x
    distance = hamming_distance(first_permutation, second_permutation)
    new_distance = 0
    iterations = 0

    while distance >= new_distance:
        iterations += 1
        first_index = int(xoroshiro128p_uniform_float64(random_states, thread_id) * dimension)
        second_index = int(xoroshiro128p_uniform_float64(random_states, thread_id) * dimension)

        tmp = second_permutation[first_index]
        second_permutation[first_index] = second_permutation[second_index]
        second_permutation[second_index] = tmp

        new_distance = hamming_distance(first_permutation, second_permutation)

        if iterations == dimension:
            break
