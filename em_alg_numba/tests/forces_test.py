import numpy as np
from main import calculate_forces
from qap.input_file_reader import InputFileReader
from em_cpu.em_solver import EMSolver
from qap.qap import QAP
from tests.test_points import test_points


if __name__ == '__main__':
    reader = InputFileReader("../../test_instances/Chr/chr20a.dat")
    dimension, weights, distances = reader.read()

    qap = QAP(weights, distances)
    qap_function = qap.get_em_function()

    solver = EMSolver(10, 20, [0] * 20, [10] * 20, qap_function)
    solver._EMSolver__points = np.array(test_points)
    calculated_forces = solver._EMSolver__calculate_forces()

    

