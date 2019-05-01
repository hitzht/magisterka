#include <iostream>

#include "ProgramArgumentsParser.h"
#include "InputFileReader.h"
#include "SolutionFileReader.h"
#include "QAP.h"
#include "PermutationFactory.h"
#include "gpu.hpp"

int main(int argc, char** argv) {
    try {
#ifndef USE_CUDA
        throw std::runtime_error{"CUDA is not enabled"};
#endif

        ProgramArgumentsParser arguments(argc, argv);

        if (arguments.hasHelp()) {
            arguments.displayHelp();
            return 0;
        }

        InputFileReader reader{arguments.getInputFile()};

        auto dimension = reader.getInstanceSize();
        auto weights = reader.getWeights();
        auto distances = reader.getDistances();

        SolutionFileReader solutionReader{arguments.getSolutionFile()};

        if (dimension != solutionReader.getInstanceSize())
            throw std::runtime_error{"Solution and input have different instance size"};

        auto solutionValue = solutionReader.getSolutionValue();
        auto solution = solutionReader.getSolution();

        QAP qap{weights, distances};

        if (qap.getValue(solution) != solutionValue)
            throw std::runtime_error("Solution value stored in file is different than calculated");

        PermutationFactory permutationFactory{};
        auto permutations = permutationFactory.get(dimension, arguments.getPopulationSize());

        auto calculatedValues = calculateOnGPU(dimension, weights, distances, permutations);

        for (unsigned i = 0; i < permutations.size(); i++) {
            if (qap.getValue(permutations[i]) != calculatedValues[i]) {
                std::cout << "diff at permutation " << i << " " << qap.getValue(permutations[i]) << " "
                          << calculatedValues[i] << std::endl;
            }
        }

    } catch (std::exception& e) {
        std::cerr << e.what() << std::endl;
    }

    return 0;
}
