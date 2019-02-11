from qap.input_file_reader import InputFileReader
from qap.solution_file_reader import SolutionFileReader
from qap.qap import QAP
import random

if __name__ == '__main__':
    input_reader = InputFileReader("../test_instances/Chr/chr20a.dat")
    lower = 0
    upper = 10

    dimension, weights, distances = input_reader.read()


    qap = QAP(weights, distances)

    input_permutation = []

    for i in range(dimension):
        input_permutation.append(random.uniform(lower, upper))

    print(input_permutation)

    f = qap.get_em_function()
    print(f(input_permutation))