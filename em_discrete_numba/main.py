import sys
import time
from tqdm import tqdm
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states
from arguments import parse_arguments
from qap.input_file_reader import InputFileReader
from qap.solution_file_reader import SolutionFileReader
from qap.qap import QAP
from em_discrete.em_discrete import generate_permutations
from em_discrete.em_discrete_cuda import em_discrete


if __name__ == '__main__':
    input_file, solution_file, permutations_count, iterations, hamming_distance = parse_arguments(sys.argv[1:])

    input_reader = InputFileReader(input_file)
    dimension, weights, distances = input_reader.read()

    qap = QAP(weights, distances)

    optimal_value = None
    optimal_permutation = None

    if solution_file is not None:
        solution_reader = SolutionFileReader(solution_file)
        solution_dimension, solution_value, solution_permutation = solution_reader.read()

        if solution_dimension != dimension:
            raise RuntimeError("Solution dimension is different than input dimension")

        if qap.get_value(solution_permutation) != solution_value:
            raise RuntimeError("Solution value does not match calculated solution permutation value")

        optimal_value = solution_value
        optimal_permutation = solution_permutation

    permutations = generate_permutations(permutations_count, dimension)
    values = [0] * permutations_count

    previous_permutations = cuda.to_device(permutations)
    next_permutations = cuda.to_device(permutations)
    pmx_buffer = cuda.to_device(permutations)

    device_values = cuda.to_device(values)
    device_weights = cuda.to_device(weights)
    device_distances = cuda.to_device(distances)

    threads_per_block = permutations_count
    blocks = 1

    random_states = create_xoroshiro128p_states(threads_per_block * blocks, seed=time.time())

    for iteration in range(iterations):
        em_discrete[blocks, threads_per_block](previous_permutations, next_permutations, device_values, device_weights,
                                               device_distances, hamming_distance, random_states, pmx_buffer)

        tmp = previous_permutations
        previous_permutations = next_permutations
        next_permutations = tmp

        host_values = device_values.copy_to_host()
        print(min(host_values))
