import numpy as np
from numba import cuda
from tests.test_points import test_points
from qap.qap import QAP
from qap.input_file_reader import InputFileReader
from em_qap_gpu import qap_device


@cuda.jit
def kernel(points, output, weights, distances):
    thread_id = cuda.threadIdx.x

    if thread_id < len(points):
        val = qap_device(points[thread_id], weights, distances)
        output[thread_id] = val


if __name__ == '__main__':
    """
    Tests if qap function for cpu and gpu returns same results
    """
    reader = InputFileReader("../../test_instances/Chr/chr20a.dat")
    dimension, weights, distances = reader.read()

    qap = QAP(weights, distances)
    qap_function = qap.get_em_function()

    output = np.zeros((len(test_points), 1))

    device_weights = cuda.to_device(weights)
    device_distances = cuda.to_device(distances)
    device_points = cuda.to_device(test_points)
    device_output = cuda.to_device(output)

    kernel[1, 10](device_points, device_output, device_weights, device_distances)

    result = device_output.copy_to_host()

    for i in range(len(test_points)):
        calculated_value = qap_function(test_points[i])
        print(i, result[i], calculated_value, result[i] == calculated_value)
