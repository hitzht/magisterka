import random
from typing import List
from em_discrete.hamming import hamming_distance


def repulsion(first_permutation: List[int], second_permutation: List[int]):
    if len(first_permutation) != len(second_permutation):
        raise RuntimeError("repulsion: permutations have different size")

    dimension = len(first_permutation)
    result = second_permutation.copy()
    distance = hamming_distance(first_permutation, second_permutation)
    new_distance = 0
    iterations = 0

    while distance >= new_distance:
        iterations += 1
        first_index = random.randint(0, dimension - 1)
        second_index = random.randint(0, dimension - 1)

        tmp = result[first_index]
        result[first_index] = result[second_index]
        result[second_index] = tmp

        new_distance = hamming_distance(first_permutation, result)

        if iterations == dimension:
            break

    return result
