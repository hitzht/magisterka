from typing import List, Tuple


class InputFileReader:
    def __init__(self, path: str) -> None:
        self.__path = path

    def read(self) -> Tuple[int, List[List[int]], List[List[int]]]:
        """
        Reads standard qap lib input file

        :return: dimension, weights matrix, distances matrix
        """
        file = open(self.__path, "r")
        file_content = [line for line in file.readlines() if len(line.strip()) > 0]
        dimension = int(file_content[0])

        values = []
        weights = []
        distances = []

        for line in file_content[1:]:
            numbers = [int(val) for val in line.split(" ") if len(val.strip()) > 0]
            values.extend(numbers)

        for i in range(dimension):
            tmp = []
            for j in range(dimension):
                tmp.append(values.pop(0))
            weights.append(tmp)

        for i in range(dimension):
            tmp = []
            for j in range(dimension):
                tmp.append(values.pop(0))
            distances.append(tmp)

        return dimension, weights, distances


if __name__ == '__main__':
    file_reader = InputFileReader("../../test_instances/Chr/chr12a.dat")
    dimension, weights, distances = file_reader.read()

    print(dimension)

    print("weights: ")
    for line in weights:
        for val in line:
            print(val, end=" ")
        print()

    print("distances:")
    for line in distances:
        for val in line:
            print(val, end=" ")
        print()
