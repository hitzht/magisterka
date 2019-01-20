from typing import List, Tuple, Callable, Any
from benchmark_functions.test_function import TestFunction


class Himmelblau(TestFunction):

    def get_name(self) -> str:
        return "Himmelblau"

    def get_function(self) -> Callable[[Any], float]:
        def himmelblau(*point) -> float:
            if len(point) == 2:
                x = point[0]
                y = point[1]
            elif len(point) == 1:
                x = point[0][0]
                y = point[0][1]
            else:
                raise ValueError("Invalid Himmelblau function argument")

            return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2

        return himmelblau

    def get_bounds(self) -> Tuple[List[float], List[float]]:
        return [-5, -5], [5, 5]

    def get_dimension(self) -> int:
        return 2

    def get_optimum_value(self) -> float:
        return 0

    def get_local_optima(self) -> List[List[float]]:
        return [[3.0, 2.0], [-2.805118, 3.131312], [-3.779310, -3.283186], [3.584428, -1.848126]]
