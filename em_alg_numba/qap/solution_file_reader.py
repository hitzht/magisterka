from typing import Tuple, List


class SolutionFileReader:
    def __init__(self, path: str) -> None:
        self.__path = path

    def read(self) -> Tuple[int, int, List[int]]:
        file = open(self.__path, "r")
        file_content = [line for line in file.readlines() if len(line.strip()) > 0]

        dimension, solution_value = [int(val) for val in file_content[0].split(" ") if len(val.strip()) > 0]
        values = []

        for line in file_content[1:]:
            values.extend([int(num) -1 for num in line.split(" ") if len(num.strip()) > 0])

        return dimension, solution_value, values


if __name__ == '__main__':
    file_reader = SolutionFileReader("../../test_instances/Chr/chr12a.out")

    dimension, solution_value, values = file_reader.read()

    print("Dimension: ", dimension)
    print("Solution value: ", solution_value)
    print("Solution: [", end=" ")

    for val in values:
        print(val, end=" ")

    print("]")

