from typing import List, Callable, Any, Tuple
from random import uniform, choice
from math import exp
import numpy as np


class EMSolver:

    def __init__(self, points_count: int, dimension: int, lower_bound: List[float], upper_bound: List[float],
                 function: Callable[[Any], float]) -> None:
        self.__points_count = points_count
        self.__dimension = dimension
        self.__lower_bound = lower_bound
        self.__upper_bound = upper_bound
        self.__function = function
        self.__points = []
        self.__validate()
        self.__initialize()

    def __validate(self):
        if self.__points_count <= 0:
            raise ValueError("Points count must be greater than zero")

        if self.__dimension <= 0:
            raise ValueError("Problem dimension must be greater than zero")

        if len(self.__lower_bound) != self.__dimension:
            raise ValueError("Dimension of lower bound ({}) is different than problem dimension({})"
                             .format(len(self.__lower_bound), self.__dimension))

        if len(self.__upper_bound) != self.__dimension:
            raise ValueError("Dimension of upper bound ({}) is different than problem dimension({})"
                             .format(len(self.__upper_bound), self.__dimension))

    def __initialize(self):
        result = []
        for _ in range(self.__points_count):
            point = []
            for bound in zip(self.__lower_bound, self.__upper_bound):
                point.append(uniform(bound[0], bound[1]))
            result.append(np.array(point))

        self.__points = result

    def find_best_point(self, points: List[np.array]) -> int:
        best_point_index = 0
        best_value = self.__function(points[0])

        for i in range(1, len(points)):
            point_value = self.__function(points[i])
            if point_value < best_value:
                best_value = point_value
                best_point_index = i

        return best_point_index

    def next_iteration(self) -> Tuple[List[np.array], List[np.array], List[np.array]]:
        self.__local_search()

        local_search_points = []
        for p in self.__points.copy():
            local_search_points.append(np.copy(p))

        forces = self.__calculate_forces()
        self.__move(forces)
        return local_search_points, forces, self.__points

    def __local_search(self):
        for point in self.__points:
            for dimension in range(self.__dimension):
                direction = choice([True, False])

                tmp_point = point.copy()
                step = uniform(0.0, 1.0)

                if direction:
                    length = self.__upper_bound[dimension] - tmp_point[dimension]
                    tmp_point[dimension] = tmp_point[dimension] + step * length
                else:
                    length = tmp_point[dimension] - self.__lower_bound[dimension]
                    tmp_point[dimension] = tmp_point[dimension] - step * length

                if self.__function(tmp_point) < self.__function(point):
                    for i in range(0, len(point)):
                        point[i] = tmp_point[i]

                    break

    def __calculate_charges(self) -> List[float]:
        charges = []

        index = self.find_best_point(self.__points)
        best_point = self.__points[index]
        best_value = self.__function(best_point)
        denominator = sum([self.__function(point) - best_value for point in self.__points])

        for point in self.__points:
            if denominator == 0:
                charge = 0
            else:
                charge = exp(-self.__dimension * (self.__function(point) - best_value) / denominator)
            charges.append(charge)

        return charges

    def __calculate_forces(self) -> List[np.array]:
        charges = self.__calculate_charges()
        forces = [np.zeros(self.__dimension)] * self.__points_count
        best_point_index = self.find_best_point(self.__points)

        for i in range(self.__points_count):
            for j in range(self.__points_count):
                if i != j and i != best_point_index:
                    distance = np.linalg.norm(self.__points[j] - self.__points[i])

                    if self.__function(self.__points[j]) < self.__function(self.__points[i]):
                        forces[i] = forces[i] + (self.__points[j] - self.__points[i]) * charges[i] * charges[j] \
                                    / (distance ** 2)
                    else:
                        forces[i] = forces[i] - (self.__points[j] - self.__points[i]) * charges[i] * charges[j] \
                                    / (distance ** 2)

        for i in range(self.__points_count):
            force_length = np.linalg.norm(forces[i])

            if force_length != 0:
                forces[i] = forces[i] / force_length

        return forces

    def __move(self, forces: List[np.array]):
        best_point_index = self.find_best_point(self.__points)

        for i in range(self.__points_count):
            if i != best_point_index:
                step = uniform(0.0, 1.0)
                force = forces[i]

                for k in range(self.__dimension):
                    # TODO tutaj jest inaczej niż w orginalnym algorytmie, dlaczego?
                    if force[k] > 0:
                        self.__points[i][k] = self.__points[i][k] + step * force[k]
                    else:
                        self.__points[i][k] = self.__points[i][k] + step * force[k]
