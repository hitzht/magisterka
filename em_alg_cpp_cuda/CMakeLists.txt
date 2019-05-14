cmake_minimum_required(VERSION 3.8)
set(CMAKE_CXX_STANDARD 17)

project(diana)

option(USE_CUDA "Use CUDA" ON)

file(GLOB cpu_source_files "${CMAKE_SOURCE_DIR}/src/cpu/*.cpp" "${CMAKE_SOURCE_DIR}/src/cpu/*.h")
file(GLOB gpu_source_files "${CMAKE_SOURCE_DIR}/src/gpu/*.cu" "${CMAKE_SOURCE_DIR}/src/gpu/*.h")

include_directories(
        libs/cxxopts/
        /usr/local/cuda-10.1/targets/x86_64-linux/include
)

enable_language("CUDA")

add_executable(diana ${cpu_source_files} ${gpu_source_files} src/main.cpp src/run.h src/run.cpp)

add_executable(run_tests ${cpu_source_files} ${gpu_source_files} src/run_tests.cpp src/run.h src/run.cpp)

# Tests
add_executable(random_cuda_test tests/random_cuda_test.cu)
add_executable(hamming_cuda_test tests/hamming_cuda_test.cu)
add_executable(pmx_cuda_test tests/pmx_cuda_test.cu)
add_executable(malloc_cuda_test tests/malloc_cuda_test.cu)
add_executable(memcpy_cuda_test tests/memcpy_cuda_test.cu)