import numpy as np
import random
from typing import List
from qap.qap import QAP
from em_discrete.hamming import hamming_distance
from em_discrete.pmx import pmx
from em_discrete.repulsion import repulsion


def generate_permutations(permutations_count, dimension):
    result = []

    start_permutation = list(range(dimension))
    for _ in range(permutations_count):
        result.append(list(np.random.permutation(start_permutation)))

    return result


def find_best_permutation(permutations, qap: QAP):
    best_permutation = permutations[0]
    best_permutation_index = 0
    best_value = qap.get_value(best_permutation)

    for i in range(len(permutations)):
        new_value = qap.get_value(permutations[i])

        if new_value < best_value:
            best_value = new_value
            best_permutation = permutations[i]
            best_permutation_index = i

    return best_permutation.copy(), best_permutation_index, best_value


def get_surroundings(source_permutation_index, permutations, distance):
    result = []

    for i in range(len(permutations)):
        if i == source_permutation_index:
            continue

        if hamming_distance(permutations[source_permutation_index], permutations[i]) <= distance:
            result.append(permutations[i].copy())

    return result


def get_random_range(upper_value):
    a = random.randint(0, upper_value)
    b = random.randint(a, upper_value)
    return a, b


def attraction_injection(permutation, surroundings, qap: QAP):
    result = permutation.copy()
    permutation_value = qap.get_value(permutation)
    dimension = len(permutation)

    for p in surroundings:
        if permutation_value >= qap.get_value(p):
            lower, upper = get_random_range(dimension - 1)
            result = pmx(p, result, lower, upper)
        else:
            result = repulsion(p, result)

    return result
