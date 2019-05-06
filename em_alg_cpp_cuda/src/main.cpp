#include <iostream>

#include "cpu/ProgramArgumentsParser.h"
#include "cpu/InputFileReader.h"
#include "cpu/SolutionFileReader.h"
#include "cpu/QAP.h"
#include "cpu/PermutationFactory.h"
#include "gpu/gpu.hpp"

int main(int argc, char** argv) {
    try {
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

        auto iterations = arguments.getIterationsCount();
        auto distance = arguments.getNeighborhoodDistance();

        auto calculatedSolutionValue = calculateOnGPU(dimension, iterations, distance, weights, distances, permutations);

        std::cout << calculatedSolutionValue << " " << solutionValue << std::endl;
    } catch (std::exception& e) {
        std::cerr << e.what() << std::endl;
    }

    return 0;
}
