import time
from numba import cuda
from em_qap_gpu import local_search, find_best_point
from tests.test_points import test_points
from qap.input_file_reader import InputFileReader
from numba.cuda.random import create_xoroshiro128p_states


if __name__ == '__main__':
    """
    Tests if in every iteration new value from local search is equal or smaller than in previous one
    """

    reader = InputFileReader("../../test_instances/Chr/chr20a.dat")
    dimension, weights, distances = reader.read()

    upper_bound = 10
    lower_bound = 0

    device_weights = cuda.to_device(weights)
    device_distances = cuda.to_device(distances)
    device_points = cuda.to_device(test_points)
    random_states = create_xoroshiro128p_states(1 * 10, seed=time.time())

    previous_value = 100000000

    for i in range(100):
        local_search[1, 10](device_points, device_weights, device_distances, upper_bound, lower_bound, random_states)
        points = device_points.copy_to_host()
        best_point, best_value, best_point_index = find_best_point(points, weights, distances)
        print(best_value)
        if previous_value < best_value:
            print("error: previous value is better than new")
            break

        previous_value = best_value
