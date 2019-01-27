from typing import List, Tuple


class InputFileReader:
    def __init__(self, path: str) -> None:
        self.__path = path

    def read(self) -> Tuple[int, List[List[int]], List[List[int]]]:
        file = open(self.__path, "r")
        file_content = [line for line in file.readlines() if len(line.strip()) > 0]
        dimension = int(file_content[0])

        weights = []
        for i in range(dimension):
            values = [int(val) for val in file_content[1 + i].split(" ") if len(val) > 0 and val != '\n']
            weights.append(values)

        distances = []
        for i in range(dimension):
            values = [int(val) for val in file_content[1 + dimension + i].split(" ") if len(val) > 0 and val != '\n']
            distances.append(values)

        file.close()

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
