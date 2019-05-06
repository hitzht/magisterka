cmake_minimum_required(VERSION 3.8)
message(STATUS "CMake version: ${CMAKE_VERSION}")

project(diana)

option(USE_CUDA "Use CUDA" ON)

file(GLOB cpu_source_files "${CMAKE_SOURCE_DIR}/src/cpu/*.cpp")
file(GLOB gpu_source_files "${CMAKE_SOURCE_DIR}/src/gpu/*.cu" "${CMAKE_SOURCE_DIR}/src/gpu/*.h")

include_directories(
        libs/cxxopts/
        /usr/local/cuda-10.1/targets/x86_64-linux/include
)

enable_language("CUDA")

add_executable(diana ${cpu_source_files} ${gpu_source_files} src/main.cpp)

# Tests
add_executable(random_cuda_test tests/random_cuda_test.cu)
add_executable(hamming_cuda_test tests/hamming_cuda_test.cu)
add_executable(pmx_cuda_test tests/pmx_cuda_test.cu)
add_executable(malloc_cuda_test tests/malloc_cuda_test.cu)
