import numpy as np
import random
import time
from typing import List
from math import exp, sqrt
from numba import cuda, int64, float64
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
from tqdm import tqdm
from qap.input_file_reader import InputFileReader
from qap.solution_file_reader import SolutionFileReader
from qap.qap import QAP


def initialize(points_count: int, dimension: int, lower_bound: float, upper_bound: float) -> List['np.array']:
    """Generates list of start points in set bounds"""

    if points_count <= 0:
        raise RuntimeError("Invalid points count")

    if dimension <= 0:
        raise RuntimeError("Invalid dimension")

    if lower_bound >= upper_bound:
        raise RuntimeError("Upper bound must be greater than lower bound")

    result = []
    for _ in range(points_count):
        point = []
        for _ in range(dimension):
            point.append(random.uniform(lower_bound, upper_bound))
        result.append(np.array(point))

    return result


@cuda.jit('void(float64[:], int64[:])', device=True)
def bubblesort(values, indexes):
    N = len(values)

    for i in range(N):
        indexes[i] = i

    tmp_values = cuda.local.array(20, float64)
    for i in range(N):
        tmp_values[i] = values[i]

    for end in range(N, 1, -1):
        for i in range(end - 1):
            cur = tmp_values[i]
            if cur > tmp_values[i + 1]:
                tmp = tmp_values[i]
                tmp_values[i] = tmp_values[i + 1]
                tmp_values[i + 1] = tmp
                t = indexes[i]
                indexes[i] = indexes[i + 1]
                indexes[i + 1] = t


@cuda.jit('int64(float64[:], int64[:,:], int64[:,:])', device=True)
def qap_device(permutation: List[float], weights: List[List[int]], distances: List[List[int]]):
    result = 0

    # TODO rozmiar tej tablicy musi byc stalą czasu kompilacji , wtf
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


def qap_host(permutation: List[float], weights: List[List[int]], distances: List[List[int]]):
    result = 0
    indexes = np.argsort(permutation)

    for a in range(len(permutation)):
        for b in range(len(permutation)):
            weight = weights[a][b]
            first_point = indexes[a]
            second_point = indexes[b]
            distance = distances[first_point][second_point]

            result += weight * distance

    return result


@cuda.jit
def local_search(points, weights, distances, upper_bound, lower_bound, random_states):
    thread_id = cuda.threadIdx.x

    if thread_id < len(points):
        pass
        # TODO rozmiar tej tablicy musi byc stalą czasu kompilacji , wtf
        tmp_point = cuda.local.array(20, float64)

        for i in range(20):
            tmp_point[i] = points[thread_id][i]

        for index in range(20):
            direction = xoroshiro128p_uniform_float32(random_states, thread_id) > 0.5
            step = xoroshiro128p_uniform_float32(random_states, thread_id)

            if direction:
                length = upper_bound - tmp_point[index]
                tmp_point[index] = tmp_point[index] + step * length
            else:
                length = tmp_point[index] - lower_bound
                tmp_point[index] = tmp_point[index] - step * length

            val1 = qap_device(tmp_point, weights, distances)
            val2 = qap_device(points[thread_id], weights, distances)
            if val1 < val2:
                for i in range(20):
                    points[thread_id][i] = tmp_point[i]
                break


@cuda.jit
def calculate_charges(points, weights, distances, best_value, denominator, charges):
    thread_id = cuda.threadIdx.x

    if thread_id < len(points):
        if denominator == 0:
            charges[thread_id] = 0
        else:
            charges[thread_id] = exp(-20 * (qap_device(points[thread_id], weights,
                                                       distances) - best_value) / denominator)


@cuda.jit('float64(float64[:], float64[:])', device=True)
def calculate_distance(first_point, second_point):
    result = 0.0

    for i in range(len(first_point)):
        result += (first_point[i] - second_point[i]) ** 2

    return sqrt(result)


@cuda.jit('float64(float64[:])', device=True)
def calculate_vector_length(vector):
    result = 0.0

    for i in range(len(vector)):
        result += (vector[i] ** 2)

    return sqrt(result)


@cuda.jit
def calculate_forces(points, weights, distances, best_point_index, charges, forces):
    thread_id = cuda.threadIdx.x

    if thread_id == best_point_index:
        return

    for j in range(len(points)):
        if thread_id != j:
            distance = calculate_distance(points[thread_id], points[j])

            # todo kurwa
            point_dif = cuda.local.array(20, float64)
            charge = (charges[thread_id][0] * charges[j][0]) / (distance ** 2)

            for i in range(20):
                point_dif[i] = (points[thread_id][i] - points[j][i]) * charge

            if qap_device(points[thread_id], weights, distances) < qap_device(points[j], weights, distances):
                for i in range(20):
                    forces[thread_id][i] += point_dif[i]
            else:
                for i in range(20):
                    forces[thread_id][i] -= point_dif[i]

    force_length = calculate_vector_length(forces[thread_id])

    if force_length != 0:
        for i in range(20):
            forces[thread_id][i] /= force_length


@cuda.jit
def move(points, best_point_index, forces, random_states):
    thread_id = cuda.threadIdx.x

    if thread_id == best_point_index:
        return

    step = xoroshiro128p_uniform_float32(random_states, thread_id)

    for k in range(20):
        points[thread_id][k] = points[thread_id][k] + step * forces[thread_id][k]


def find_best_point(points, weights, distances):
    best_point = points[0]
    best_value = qap_host(best_point, weights, distances)
    best_index = 0

    for i in range(len(points)):
        new_value = qap_host(points[i], weights, distances)
        if new_value < best_value:
            best_point = points[i]
            best_value = new_value
            best_index = i

    return best_point, best_value, best_index


def solve(input_file, solution_file, points_count, iterations, upper_bound, lower_bound, show_progress=False):
    input_reader = InputFileReader(input_file)
    dimension, weights, distances = input_reader.read()

    optimal_value = None
    optimal_permutation = None

    if solution_file is not None:
        solution_reader = SolutionFileReader(solution_file)
        solution_dimension, solution_value, solution_permutation = solution_reader.read()

        qap = QAP(weights, distances)

        if solution_dimension != dimension:
            raise RuntimeError("Solution dimension is different than input dimension")

        if qap.get_value(solution_permutation) != solution_value:
            raise RuntimeError("Solution value does not match calculated solution permutation value")

        optimal_value = solution_value
        optimal_permutation = solution_permutation

    random.seed(time.time())
    start_points = initialize(points_count, dimension, lower_bound, upper_bound)

    r = tqdm(range(iterations)) if show_progress else range(iterations)

    device_points = cuda.to_device(start_points)
    device_weights = cuda.to_device(weights)
    device_distances = cuda.to_device(distances)

    charges = np.zeros((points_count, 1))
    forces = np.zeros((points_count, dimension))
    device_charges = cuda.to_device(charges)
    device_forces = cuda.to_device(forces)

    # TODO: add check if points_count is lower than max threads per block
    threads_per_block = points_count
    blocks = 1

    random_states = create_xoroshiro128p_states(threads_per_block * blocks, seed=time.time())

    best_values = []

    for it in r:
        local_search[blocks, threads_per_block](device_points, device_weights, device_distances,
                                                upper_bound, lower_bound, random_states)

        points = device_points.copy_to_host()
        best_point, best_value, best_index = find_best_point(points, weights, distances)
        denominator = sum([qap_host(point, weights, distances) - best_value for point in points])

        calculate_charges[blocks, threads_per_block](device_points, device_weights, device_distances, best_value,
                                                     denominator, device_charges)

        calculate_forces[blocks, threads_per_block](device_points, device_weights, device_distances, best_index,
                                                    device_charges, device_forces)

        move[blocks, threads_per_block](device_points, best_index, device_forces, random_states)

        points = device_points.copy_to_host()
        best_point, best_value, best_index = find_best_point(points, weights, distances)

        best_values.append(best_value)

    return best_values, optimal_value
#
# if __name__ == '__main__':
#     input_reader = InputFileReader("../test_instances/Chr/chr20a.dat")
#     lower_bound = 0
#     upper_bound = 10
#     points_count = 100
#     iterations = 1000
#
#     dimension, weights, distances = input_reader.read()
#
#     random.seed(time.time())
#     start_points = initialize(points_count, dimension, lower_bound, upper_bound)
#
#     device_points = cuda.to_device(start_points)
#     device_weights = cuda.to_device(weights)
#     device_distances = cuda.to_device(distances)
#
#     threads_per_block = points_count
#     blocks = 1
#
#     random_states = create_xoroshiro128p_states(threads_per_block * blocks, seed=time.time())
#
#     for i in range(iterations):
#         print("iteration", i)
#         charges = np.zeros((points_count, 1))
#         forces = np.zeros((points_count, dimension))
#         device_charges = cuda.to_device(charges)
#         device_forces = cuda.to_device(forces)
#
#
#         local_search[blocks, threads_per_block](device_points, device_weights, device_distances,
#                                                 upper_bound, lower_bound, random_states)
#
#         points = device_points.copy_to_host()
#         best_point, best_value, best_index = find_best_point(points, weights, distances)
#         denominator = sum([qap_host(point, weights, distances) - best_value for point in points])
#
#         calculate_charges[blocks, threads_per_block](device_points, device_weights, device_distances, best_value,
#                                                      denominator, device_charges)
#
#         calculate_forces[blocks, threads_per_block](device_points, device_weights, device_distances, best_index,
#                                                     device_charges, device_forces)
#
#         move[blocks, threads_per_block](device_points, best_index, device_forces, random_states)
#
#         points = device_points.copy_to_host()
#         best_point, best_value, best_index = find_best_point(points, weights, distances)
#         print(best_value)
