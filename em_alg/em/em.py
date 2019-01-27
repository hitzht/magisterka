from random import uniform, choice
from typing import List, Callable, Any
from math import exp
import numpy as np


def initialize(points_count: int, lower_bound: List[float], upper_bound: List[float]) -> List['np.array']:
    if len(lower_bound) != len(upper_bound):
        raise RuntimeError("Lower bound and upper bound have different size")

    if points_count <= 0:
        raise RuntimeError("Invalid points count")

    result = []
    for _ in range(points_count):
        point = []
        for bound in zip(lower_bound, upper_bound):
            point.append(uniform(bound[0], bound[1]))
        result.append(np.array(point))

    return result


def find_best_point(points: List['np.array'], function: Callable[[List[float]], float]) -> int:
    best_point_index = 0
    best_value = function(points[0])

    for i in range(1, len(points)):
        point_value = function(points[i])
        if point_value < best_value:
            best_value = point_value
            best_point_index = i

    return best_point_index


def local_search(iterations: int, factor: float, lower_bound: List[float], upper_bound: List[float],
                 points: List['np.array'], function: Callable[[List[float]], float]) -> List[List[float]]:
    if iterations <= 0:
        raise RuntimeError("Invalid local search iterations value")

    if not 0.0 <= factor <= 1:
        raise RuntimeError("Invalid local search factor value")


    length = factor * max([bound[1] - bound[0] for bound in zip(lower_bound, upper_bound)])
    problem_size = len(lower_bound)

    for point in points:
        for dimension in range(0, problem_size):
            direction = choice([True, False])

            counter = 1
            while counter < iterations:
                tmp_point = point.copy()
                step = uniform(0.0, 1.0)

                if direction:
                    tmp_point[dimension] = tmp_point[dimension] + step * length
                else:
                    tmp_point[dimension] = tmp_point[dimension] - step * length

                if function(tmp_point) < function(point):
                    counter = iterations - 1
                    for i in range(0, len(point)):
                        point[i] = tmp_point[i]

                counter += 1

    return points


def calculate_charges(points: List['np.array'], function: Callable[[List[float]], float]) -> List[float]:
    charges = []

    best_point = min(points, key=function)
    dimension = len(best_point)
    denominator = sum([function(point) - function(best_point) for point in points])

    for point in points:
        charge = exp(-dimension * (function(point) - function(best_point)) / denominator)
        charges.append(charge)

    return charges


def calculate_forces(points: List['np.array'], function: Callable[[Any], float]) -> List['np.array']:
    charges = calculate_charges(points, function)
    dimension = len(points[0])
    forces = [np.zeros(dimension)] * len(points)

    for i in range(0, len(points)):
        for j in range(0, len(points)):
            if i != j:
                distance = np.linalg.norm(points[j] - points[i])
                if function(points[j]) < function(points[j]):
                    forces[i] = forces[i] + (points[j] - points[i]) * charges[i] * charges[j] / (distance ** 2)
                else:
                    forces[i] = forces[i] - (points[j] - points[i]) * charges[i] * charges[j] / (distance ** 2)

    return forces


def move(points: List['np.array'], forces: List['np.array'], lower_bound: List[float], upper_bound: List[float], function: Callable[[List[float]], float]):
    best_point_index = find_best_point(points, function)

    for i in range(len(points)):
        if i != best_point_index:
            step = uniform(0.0, 1.0)
            force = forces[i] / np.linalg.norm(forces[i])

            for k in range(len(points[i])):
                if force[k] > 0:
                    points[i][k] = points[i][k] + step * force[k] * (upper_bound[k] - points[i][k])
                else:
                    points[i][k] = points[i][k] + step * force[k] * (points[i][k] - lower_bound[k])

    return points
