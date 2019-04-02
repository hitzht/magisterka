import numpy as np


def generate_permutations(permutations_count, dimension):
    result = []

    start_permutation = list(range(dimension))
    for _ in range(permutations_count):
        result.append(list(np.random.permutation(start_permutation)))

    return result
