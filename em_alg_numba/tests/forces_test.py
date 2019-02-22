import numpy as np
from numba import cuda
from em_qap_gpu import calculate_forces, find_best_point, calculate_charges, qap_host
from qap.input_file_reader import InputFileReader
from em_cpu.em_solver import EMSolver
from qap.qap import QAP
from tests.test_points import test_points


if __name__ == '__main__':
    """
    Checks if forces calculated by cpu and gpu versions are same
    """
    reader = InputFileReader("../../test_instances/Chr/chr20a.dat")
    dimension, weights, distances = reader.read()

    qap = QAP(weights, distances)
    qap_function = qap.get_em_function()

    solver = EMSolver(10, 20, [0] * 20, [10] * 20, qap_function)
    solver._EMSolver__points = np.array(test_points)
    calculated_forces = solver._EMSolver__calculate_forces()

    charges = np.zeros((len(test_points), 1))
    forces = np.zeros((len(test_points), 20))

    device_points = cuda.to_device(test_points)
    device_weights = cuda.to_device(weights)
    device_distances = cuda.to_device(distances)
    device_charges = cuda.to_device(charges)
    device_forces = cuda.to_device(forces)

    best_point, best_value, best_index = find_best_point(test_points, weights, distances)
    denominator = sum([qap_host(point, weights, distances) - best_value for point in test_points])
    calculate_charges[1, 10](device_points, device_weights, device_distances, best_value, denominator, device_charges)

    calculate_forces[1, 10](device_points, device_weights, device_distances, best_index,
                            device_charges, device_forces)

    forces = device_forces.copy_to_host()
    for i in range(len(forces)):
        print(forces[i])
        print(calculated_forces[i])
        print(forces[i] == calculated_forces[i])
