from typing import List, Callable, Any, Tuple


class TestFunction:
    def get_name(self) -> str:
        raise NotImplementedError()

    def get_function(self) -> Callable[[Any], float]:
        raise NotImplementedError()

    def get_bounds(self) -> Tuple[List[float], List[float]]:
        raise NotImplementedError()

    def get_dimension(self) -> int:
        raise NotImplementedError()

    def get_optimum_value(self) -> float:
        raise NotImplementedError()

    def get_local_optima(self) -> List[List[float]]:
        raise NotImplementedError()

