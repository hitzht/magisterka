cmake_minimum_required(VERSION 3.12)
project(qap_cpu)

set(CMAKE_CXX_STANDARD 17)

include(${PROJECT_SOURCE_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

add_executable(qap_cpu
        src/main.cpp
        src/QAP.cpp
        src/QAP.h
        src/InputFileReader.cpp
        src/InputFileReader.h
        src/QAPDataTypes.h
        src/EMAlgorithm.cpp
        src/EMAlgorithm.h
        src/QAPInterface.h
        src/PermutationFactory.cpp
        src/PermutationFactory.h
        src/HammingDistance.cpp
        src/HammingDistance.h
        src/PMX.cpp
        src/PMX.h
        src/RangeGenerator.cpp
        src/RangeGenerator.h)

#add_executable(unit_tests
#        unit_tests/main.cpp
#        src/HammingDistance.cpp
#        unit_tests/HammingDistanceTests.cpp
#        src/PMX.cpp
#        unit_tests/PMXTests.cpp)

# target_link_libraries(unit_tests gtest gtest_main)