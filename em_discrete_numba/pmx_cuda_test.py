from numba import cuda
from em_discrete.pmx_cuda import pmx


def test1():
    p1 = [8, 4, 7, 3, 6, 2, 5, 1, 9, 0]
    p2 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    p1_device = cuda.to_device(p1)
    p2_device = cuda.to_device(p2)
    result_device = cuda.to_device(p1)

    pmx[1, 1](p1_device, p2_device, 3, 7, result_device)

    print(p1_device.copy_to_host() == [8, 4, 7, 3, 6, 2, 5, 1, 9, 0])
    print(p2_device.copy_to_host() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    print(result_device.copy_to_host() == [0, 7, 4, 3, 6, 2, 5, 1, 8, 9])


def test2():
    p1 = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    p2 = [4, 5, 2, 1, 8, 7, 6, 9, 3]

    p1_device = cuda.to_device(p1)
    p2_device = cuda.to_device(p2)
    result_device = cuda.to_device(p1)

    pmx[1, 1](p1_device, p2_device, 3, 6, result_device)
    print(p1_device.copy_to_host() == [1, 2, 3, 4, 5, 6, 7, 8, 9])
    print(p2_device.copy_to_host() == [4, 5, 2, 1, 8, 7, 6, 9, 3])
    print(result_device.copy_to_host() == [1, 8, 2, 4, 5, 6, 7, 9, 3])


def test3():
    p1 = [1, 5, 2, 8, 7, 4, 3, 6]
    p2 = [4, 2, 5, 8, 1, 3, 6, 7]

    p1_device = cuda.to_device(p1)
    p2_device = cuda.to_device(p2)
    result_device = cuda.to_device(p1)

    pmx[1, 1](p1_device, p2_device, 2, 4, result_device)
    print(p1_device.copy_to_host() == [1, 5, 2, 8, 7, 4, 3, 6])
    print(p2_device.copy_to_host() == [4, 2, 5, 8, 1, 3, 6, 7])
    print(result_device.copy_to_host() == [4, 5, 2, 8, 7, 3, 6, 1])


if __name__ == '__main__':
    test1()
    test2()
    test3()

