from typing import List
from numba import cuda


@cuda.jit('int64(int64[:], int64)', device=True)
def find_value_index_in_array(array, value):
    for i in range(len(array)):
        if array[i] == value:
            return i

    return -1


@cuda.jit('void(int64, int64, int64[:], int64[:], int64[:], int64, int64)', device=True)
def switch_value(value_to_switch: int, value_to_write: int, first_permutation: List[int], second_permutation: List[int],
                 result_permutation: List[int], lower: int, upper: int):
    while True:
        index_in_second_parent = find_value_index_in_array(second_permutation, value_to_switch)
        value_from_first_parent = first_permutation[index_in_second_parent]
        index_in_second_parent = find_value_index_in_array(second_permutation, value_from_first_parent)

        if lower <= index_in_second_parent <= upper:
            value_to_switch = value_from_first_parent
        else:
            result_permutation[index_in_second_parent] = value_to_write
            break


@cuda.jit('boolean(int64, int64[:], int64, int64)', device=True)
def value_is_in_range(value, permutation, lower, upper):
    for i in range(lower, upper + 1):
        if permutation[i] == value:
            return True

    return False


@cuda.jit('void(int64[:], int64[:], int64, int64, int64[:])', device=True)
def pmx(first_permutation, second_permutation, lower, upper, result):
    for i in range(len(first_permutation)):
        result[i] = -1

    for i in range(lower, upper+1):
        result[i] = first_permutation[i]

    for i in range(lower, upper + 1):
        value = second_permutation[i]
        if not value_is_in_range(value, first_permutation, lower, upper):
            switch_value(value, value, first_permutation, second_permutation, result, lower, upper)

    for i in range(len(first_permutation)):
        if result[i] == -1:
            result[i] = second_permutation[i]
