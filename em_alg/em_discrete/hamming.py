from typing import List


def hamming_distance(first_permutation: List[int], second_permutation: List[int]):
    if len(first_permutation) != len(second_permutation):
        raise RuntimeError("Permutations have different size")

    result = 0
    for i in range(len(first_permutation)):
        if first_permutation[i] != second_permutation[i]:
            result += 1

    return result
