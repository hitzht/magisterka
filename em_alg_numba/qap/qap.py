from typing import List, Callable, Any
import numpy as np


class QAP:
    def __init__(self, weights: List[List[int]], distances: List[List[int]]) -> None:
        self.__weights = weights
        self.__distances = distances

    def get_value(self, permutation: List[int]) -> int:
        result = 0

        for a in range(len(permutation)):
            for b in range(len(permutation)):
                weight = self.__weights[a][b]
                first_point = permutation[a]
                second_point = permutation[b]
                distance = self.__distances[first_point][second_point]

                result += weight * distance

        return result

    def get_em_function(self) -> Callable[[List[Any]], int]:
        def f(permutation: List[int]):
            sorted_indexes = np.argsort(permutation)
            return self.get_value(sorted_indexes)

        return f
