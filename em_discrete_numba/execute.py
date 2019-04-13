import time
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states
from qap.input_file_reader import InputFileReader
from qap.solution_file_reader import SolutionFileReader
from qap.qap import QAP
from em_discrete.em_discrete import generate_permutations
from em_discrete.em_discrete_cuda import em_discrete


def execute(input_file, solution_file, permutations_count, iterations, hamming_distance_factor):
    #input_file, solution_file, permutations_count, iterations, hamming_distance = parse_arguments(sys.argv[1:])

    input_reader = InputFileReader(input_file)
    dimension, weights, distances = input_reader.read()

    hamming_distance = int(dimension * hamming_distance_factor)

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

    start_time1 = time.time()

    previous_permutations = cuda.to_device(permutations)
    next_permutations = cuda.to_device(permutations)
    pmx_buffer = cuda.to_device(permutations)

    device_values = cuda.to_device(values)
    device_weights = cuda.to_device(weights)
    device_distances = cuda.to_device(distances)

    threads_per_block = permutations_count
    blocks = 1

    random_states = create_xoroshiro128p_states(threads_per_block * blocks, seed=time.time())

    start_time2 = time.time()

    for _ in range(iterations):
        em_discrete[blocks, threads_per_block](previous_permutations, next_permutations, device_values, device_weights,
                                               device_distances, hamming_distance, random_states, pmx_buffer)

        tmp = previous_permutations
        previous_permutations = next_permutations
        next_permutations = tmp

    end_time2 = time.time()

    host_values = device_values.copy_to_host()
    end_time1 = time.time()

    best = min(host_values)
    diff = (best - optimal_value)/best * 100

    return best, round(diff, 2), round(end_time2 - start_time2, 2), round(end_time1 - start_time1, 2)

