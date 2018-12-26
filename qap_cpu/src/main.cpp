#include <iostream>
#include "InputFileReader.h"
#include "QAP.h"
#include "EMAlgorithm.h"
#include "ProgramArgumentsParser.h"

void displayPermutation(const Permutation<unsigned>& permutation) {
    for (const auto& elem : permutation)
        std::cout << elem << " ";

    std::cout << std::endl;
}

int main(int argc, char** argv) {
    try {
        ProgramArgumentsParser arguments(argc, argv);

        InputFileReader reader{arguments.getInputFile()};

        auto instanceSize = reader.getInstanceSize();
        auto weights = reader.getWeights();
        auto distances = reader.getDistances();

        QAP qap{weights, distances};
        EMAlgorithm algorithm{qap};

        auto populationSize = arguments.getPopulationSize();
        auto iterationsCount = arguments.getIterationsCount();
        auto neighborhoodDistance = arguments.getNeighborhoodDistance();

        auto solution = algorithm.solve(instanceSize, populationSize, iterationsCount, neighborhoodDistance);

        displayPermutation(solution);

        if (arguments.hasSolutionFile()) {

        }
    } catch (std::exception& e) {
        std::cerr << e.what() << std::endl;
    }

    return 0;
}
