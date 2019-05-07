#include <iostream>

#include "cpu/ProgramArgumentsParser.h"
#include "cpu/CUDAUtils.h"
#include "cpu/InputFileReader.h"
#include "cpu/SolutionFileReader.h"
#include "cpu/QAP.h"
#include "cpu/PermutationFactory.h"
#include "cpu/AlgorithmInput.h"
#include "gpu/EMAlgorithm.h"

int main(int argc, char** argv) {
    try {
        ProgramArgumentsParser arguments(argc, argv);

        if (arguments.hasHelp()) {
            arguments.displayHelp();
            return 0;
        }

        auto blocks = arguments.getBlocks();
        auto populationPerBlock = arguments.getPopulationSize();

        CUDAUtils::checkParameters(blocks, populationPerBlock);

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
        auto permutations = permutationFactory.get(dimension, blocks * populationPerBlock);

        AlgorithmInput algorithmInput{};
        algorithmInput.blocks = blocks;
        algorithmInput.threads = populationPerBlock;
        algorithmInput.dimension = dimension;
        algorithmInput.iterations = arguments.getIterationsCount();
        algorithmInput.neighborhoodDistance = arguments.getNeighborhoodDistance();
        algorithmInput.weights = std::move(weights);
        algorithmInput.distances = std::move(distances);
        algorithmInput.permutations = std::move(permutations);

        EMAlgorithm algorithm{};
        auto calculatedSolutionValue = algorithm.solve(algorithmInput);

        auto diff = double(calculatedSolutionValue)/double(solutionValue) * 100 - 100;
        std::cout << calculatedSolutionValue << " " << solutionValue << " " << diff << std::endl;
    } catch (std::exception& e) {
        std::cerr << e.what() << std::endl;
    }

    return 0;
}
