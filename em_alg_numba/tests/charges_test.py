import numpy as np
from numba import cuda
from qap.input_file_reader import InputFileReader
from em_cpu.em_solver import EMSolver
from tests.test_points import test_points
from qap.qap import QAP
from main import calculate_charges, find_best_point, qap_host


if __name__ == '__main__':
    """
    Tests if calculate charge function for cpu and gpu returns same values 
    """
    reader = InputFileReader("../../test_instances/Chr/chr20a.dat")
    dimension, weights, distances = reader.read()

    qap = QAP(weights, distances)
    qap_function = qap.get_em_function()

    solver = EMSolver(len(test_points), 20, [0] * 20, [10] * 20, qap_function)
    solver._EMSolver__points = test_points
    calculated_charges = solver._EMSolver__calculate_charges()

    charges = np.zeros((len(test_points), 1))

    device_points = cuda.to_device(test_points)
    device_weights = cuda.to_device(weights)
    device_distances = cuda.to_device(distances)
    device_charges = cuda.to_device(charges)

    best_point, best_value, best_index = find_best_point(test_points, weights, distances)
    denominator = sum([qap_host(point, weights, distances) - best_value for point in test_points])
    calculate_charges[1, 10](device_points, device_weights, device_distances, best_value, denominator, device_charges)
    ch = device_charges.copy_to_host()

    for i in range(10):
        print(calculated_charges[i], ch[i], calculated_charges[i] == ch[i])
