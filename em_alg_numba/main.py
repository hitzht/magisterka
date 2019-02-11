import numpy as np
import random
from input_file_reader import InputFileReader
from numba import cuda, int64
from typing import List


def initialize(points_count: int, lower_bound: float, upper_bound: float) -> List['np.array']:
    """Generates list of start points in set bounds"""

    if lower_bound >= upper_bound:
        raise RuntimeError("Upper bound must be greater than lower bound")

    if points_count <= 0:
        raise RuntimeError("Invalid points count")

    result = []
    for _ in range(points_count):
        point = []
        for bound in zip(lower_bound, upper_bound):
            point.append(random.uniform(bound[0], bound[1]))
        result.append(np.array(point))

    return result


@cuda.jit('void(float64[:], int64[:])', device=True)
def bubblesort(values, indexes):
    N = len(values)

    for i in range(N):
        indexes[i] = i

    for end in range(N, 1, -1):
        for i in range(end - 1):
            cur = values[i]
            if cur > values[i + 1]:
                tmp = values[i]
                values[i] = values[i + 1]
                values[i + 1] = tmp
                t = indexes[i]
                indexes[i] = indexes[i + 1]
                indexes[i + 1] = t


@cuda.jit('int64(float64[:], int64[:,:], int64[:,:])', device=True)
def qap(permutation: List[float], weights: List[List[int]], distances: List[List[int]]):
    result = 0

    indexes = cuda.local.array(20, int64)
    bubblesort(permutation, indexes)

    for a in range(len(permutation)):
        for b in range(len(permutation)):
            weight = weights[a][b]
            first_point = indexes[a]
            second_point = indexes[b]
            distance = distances[first_point][second_point]

            result += weight * distance

    return result


@cuda.jit
def local_search(points, weights, distances):
    pass


@cuda.jit
def kernel(permutation, weights, distances, result):
    result[0] = qap(permutation, weights, distances)


if __name__ == '__main__':
    input_reader = InputFileReader("../test_instances/Chr/chr20a.dat")
    lower = 0
    upper = 10
    points = 10

    dimension, weights, distances = input_reader.read()
    start_points = initialize(points, lower, upper)

    device_weights = cuda.to_device(weights)
    device_distances = cuda.to_device(distances)
    device_points = cuda.to_device(start_points)


    input_permutation = [0.19497359652297264, 7.630297011269675, 0.5847167073774862, 3.3412885270815105, 4.752704294067644, 0.4112064496003598, 0.6494208855356975, 1.3257172770057168, 8.505545702236283, 1.5945965077875845, 2.141845559882004, 2.1248632265971965, 8.778836740526861, 8.816840766282924, 3.5123967188886676, 6.150677000907484, 5.800047523427003, 9.332530217725306, 9.469558726297693, 1.2692515832178342]

    # for i in range(dimension):
    #     input_permutation.append(random.uniform(lower, upper))

    print(input_permutation)
    device_permutation = cuda.to_device(input_permutation)
    res = cuda.to_device((1))

    kernel[1, 1](device_permutation, device_weights, device_distances, res)
    ddd = res.copy_to_host()
    print(ddd)

