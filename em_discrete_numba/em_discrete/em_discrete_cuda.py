from numba import cuda
from numba.cuda.random import xoroshiro128p_uniform_float64
from typing import List
from em_discrete.hamming_cuda import hamming_distance
from em_discrete.pmx_cuda import pmx
from em_discrete.repulsion_cuda import repulsion


@cuda.jit('int64(int64[:], int64[:,:], int64[:,:])', device=True)
def qap_device(permutation: List[int], weights: List[List[int]], distances: List[List[int]]):
    result = 0

    for a in range(len(permutation)):
        for b in range(len(permutation)):
            weight = weights[a][b]
            first_point = permutation[a]
            second_point = permutation[b]
            distance = distances[first_point][second_point]

            result += weight * distance

    return result


@cuda.jit
def em_discrete(previous_permutations: List[List[int]], next_permutations: List[List[int]], qap_values: List[int],
                weights: List[List[int]], distances: List[List[int]], max_hamming_distance: int, random_states,
                pmx_buffer):
    thread_id = cuda.threadIdx.x

    if thread_id < len(previous_permutations):
        dimension = len(previous_permutations[thread_id])
        qap_values[thread_id] = qap_device(previous_permutations[thread_id], weights, distances)

        cuda.syncthreads()

        best_value_index = 0

        for i in range(len(previous_permutations)):
            if qap_values[best_value_index] > qap_values[i]:
                best_value_index = i

        cuda.syncthreads()

        # copy current permutation to next permutation
        for i in range(dimension):
            next_permutations[thread_id][i] = previous_permutations[thread_id][i]

        if thread_id == best_value_index:
            return

        # search surroundings
        for i in range(len(previous_permutations)):
            if i == thread_id:
                continue

            if hamming_distance(previous_permutations[thread_id], previous_permutations[thread_id]) < max_hamming_distance:
                if qap_values[thread_id] > qap_values[i]:
                    first_bound = int(xoroshiro128p_uniform_float64(random_states, thread_id))
                    second_bound = int(xoroshiro128p_uniform_float64(random_states, thread_id))
                    lower_bound = min(first_bound, second_bound)
                    upper_bound = max(first_bound, second_bound)

                    pmx(previous_permutations[i], next_permutations[thread_id], lower_bound, upper_bound,
                        pmx_buffer[thread_id])

                    for j in range(dimension):
                        next_permutations[thread_id][j] = pmx_buffer[thread_id][j]

                else:
                    repulsion(previous_permutations[i], next_permutations[thread_id], random_states)
