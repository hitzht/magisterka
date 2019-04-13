import time
from qap.input_file_reader import InputFileReader
from qap.solution_file_reader import SolutionFileReader
from qap.qap import QAP
from em_discrete.em_discrete import generate_permutations, find_best_permutation, get_surroundings, attraction_injection


def execute(input_file, solution_file, permutations_count, iterations, distance_factor):
    input_reader = InputFileReader(input_file)
    dimension, weights, distances = input_reader.read()
    distance = dimension * distance_factor

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

    start_time = time.time()

    for iteration in range(iterations):
        best_permutation, best_permutation_index, best_value = find_best_permutation(permutations, qap)

        next_permutations = []

        for i in range(len(permutations)):
            if i == best_permutation_index:
                next_permutations.append(permutations[i])
                continue
            else:
                surroundings = get_surroundings(i, permutations, distance)
                new_permutation = attraction_injection(permutations[i], surroundings, qap)
                next_permutations.append(new_permutation)

        permutations = next_permutations

    end_time = time.time()

    values = [qap.get_value(p) for p in permutations]
    best = min(values)
    diff = (best - optimal_value) / best * 100

    return best, round(diff, 2), round(end_time - start_time, 2)
