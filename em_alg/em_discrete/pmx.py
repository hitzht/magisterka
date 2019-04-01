from typing import List


def pmx(first_permutation, second_permutation, lower, upper):
    if len(first_permutation) != len(second_permutation):
        raise RuntimeError("pmx: permutations have different size")

    if not 0 <= lower < len(first_permutation):
        raise RuntimeError("pmx: lower bound is too big ", lower)

    if not 0 <= upper < len(first_permutation):
        raise RuntimeError("pmx: upper bound is too big ", upper)

    if lower > upper:
        raise RuntimeError("pmx: lower bound is greater than upper")

    result = [-1] * len(first_permutation)
    result[lower:upper+1] = first_permutation[lower:upper+1]

    tmp = []
    for i in range(lower, upper + 1):
        if not second_permutation[i] in first_permutation[lower:upper+1]:
            tmp.append(second_permutation[i])

    for value in tmp:
        switch_value(value, value, first_permutation, second_permutation, result, lower, upper)

    for i in range(len(first_permutation)):
        if result[i] == -1:
            result[i] = second_permutation[i]

    return result


def switch_value(value_to_switch: int, value_to_write: int, first_permutation: List[int], second_permutation: List[int],
                 result_permutation: List[int], lower: int, upper: int):
    index_in_second_parent = second_permutation.index(value_to_switch)
    value_from_first_parent = first_permutation[index_in_second_parent]
    index_in_second_parent = second_permutation.index(value_from_first_parent)

    if lower <= index_in_second_parent <= upper:
        switch_value(value_from_first_parent, value_to_write, first_permutation, second_permutation, result_permutation,
                     lower, upper)
    else:
        result_permutation[index_in_second_parent] = value_to_write
