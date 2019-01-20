from typing import Dict, Callable
from benchmark_functions.test_function import TestFunction
from benchmark_functions.himmelblau import Himmelblau
from benchmark_functions.ackley import Ackley


class BenchmarkFactory:

    def __init__(self) -> None:
        self.__functions = {
            "himmelblau": Himmelblau(),
            "ackley": Ackley()
        }

    def get_benchmark_function(self, function_name: str) -> TestFunction:
        if function_name.lower() in map(lambda s: s.lower(), self.__functions.keys()):
            return self.__functions[function_name]
        else:
            raise ValueError("Could not find function {}".format(function_name))

    def get_all_functions(self) -> Dict[str, TestFunction]:
        return self.__functions
