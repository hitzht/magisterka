from numba import cuda

if __name__ == '__main__':
    gpu = cuda.get_current_device()
    print("name = %s" % gpu.name)
    print("max threads per block & %s \\\\" % str(gpu.MAX_THREADS_PER_BLOCK))
    print("maxBlockDimX & %s \\\\" % str(gpu.MAX_BLOCK_DIM_X))
    print("maxBlockDimY & %s \\\\" % str(gpu.MAX_BLOCK_DIM_Y))
    print("maxBlockDimZ & %s \\\\" % str(gpu.MAX_BLOCK_DIM_Z))
    print("maxGridDimX & %s \\\\" % str(gpu.MAX_GRID_DIM_X))
    print("maxGridDimY & %s \\\\" % str(gpu.MAX_GRID_DIM_Y))
    print("maxGridDimZ & %s \\\\" % str(gpu.MAX_GRID_DIM_Z))
    print("maxSharedMemoryPerBlock & %s \\\\" % str(gpu.MAX_SHARED_MEMORY_PER_BLOCK))
    print("asyncEngineCount & %s \\\\" % str(gpu.ASYNC_ENGINE_COUNT))
    print("canMapHostMemory & %s \\\\" % str(gpu.CAN_MAP_HOST_MEMORY))
    print("multiProcessorCount & %s \\\\" % str(gpu.MULTIPROCESSOR_COUNT))
    print("warpSize & %s \\\\" % str(gpu.WARP_SIZE))
    print("unifiedAddressing & %s \\\\" % str(gpu.UNIFIED_ADDRESSING))
    print(gpu.compute_capability)

