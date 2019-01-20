from typing import List, Tuple, Callable, Any
from benchmark_functions.test_function import TestFunction
from math import exp, sqrt, cos, pi, e
import numpy as np


class Ackley(TestFunction):
    def get_name(self) -> str:
        return "Ackley"

    def get_function(self) -> Callable[[Any], float]:
        def ackley(*point):
            if len(point) == 2:
                x = point[0]
                y = point[1]
            elif len(point) == 1:
                x = point[0][0]
                y = point[0][1]
            else:
                raise ValueError("Invalid Ackley function argument")

            return -20 * np.exp(-0.2 * np.sqrt(0.5 * (x ** 2 + y ** 2))) - np.exp(0.5 * (np.cos(2 * pi * x) + np.cos(2 * pi * y))) + e + 20

        return ackley

    def get_bounds(self) -> Tuple[List[float], List[float]]:
        return [-5, -5], [5, 5]

    def get_dimension(self) -> int:
        return 2

    def get_optimum_value(self) -> float:
        return 0

    def get_local_optima(self) -> List[List[float]]:
        return [[0, 0]]