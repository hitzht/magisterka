cmake_minimum_required(VERSION 3.12)
project(qap_cpu)

set(CMAKE_CXX_STANDARD 17)

include(${PROJECT_SOURCE_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

add_executable(qap_cpu
        main.cpp
        QAP.cpp
        QAP.h
        InputFileReader.cpp
        InputFileReader.h
        QAPDataTypes.h)

add_executable(unit_tests
        unit_tests/main.cpp)

target_link_libraries(unit_tests gtest gtest_main)