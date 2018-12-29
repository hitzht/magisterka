#include <iostream>
#include <iomanip>
#include "InputFileReader.h"
#include "QAP.h"
#include "EMAlgorithm.h"
#include "ProgramArgumentsParser.h"
#include "SolutionFileReader.h"

void displayPermutation(const Permutation<unsigned>& permutation) {
    for (const auto& elem : permutation)
        std::cout << elem << " ";

    std::cout << std::endl;
}

int main(int argc, char** argv) {
    try {
        ProgramArgumentsParser arguments(argc, argv);

        if (arguments.hasHelp()) {
            arguments.displayHelp();
            return 0;
        }

        InputFileReader reader{arguments.getInputFile()};

        auto instanceSize = reader.getInstanceSize();
        auto weights = reader.getWeights();
        auto distances = reader.getDistances();

        QAP qap{weights, distances};
        EMAlgorithm algorithm{qap};

        auto populationSize = arguments.getPopulationSize();
        auto iterationsCount = arguments.getIterationsCount();
        auto neighborhoodDistance = arguments.getNeighborhoodDistance();

        auto calculatedSolution = algorithm.solve(instanceSize, populationSize, iterationsCount, neighborhoodDistance);
        auto calculatedSolutionValue = qap.getValue(calculatedSolution);

        if (arguments.hasSolutionFile()) {
            SolutionFileReader solutionReader{arguments.getSolutionFile()};

            if (instanceSize != solutionReader.getInstanceSize())
                throw std::runtime_error{"Solution and input have different instance size"};

            auto originalSolutionValue = solutionReader.getSolutionValue();
            auto originalSolution = solutionReader.getSolution();

            if (qap.getValue(originalSolution) != originalSolutionValue)
                throw std::runtime_error("Solution value stored in file is different than calculated");

            std::cout << calculatedSolutionValue << " ";
            std::cout << originalSolutionValue << " ";
            std::cout << std::fixed << std::setprecision(2);
            std::cout << double(calculatedSolutionValue)/originalSolutionValue * 100 << std::endl;
        } else {
            displayPermutation(calculatedSolution);
            std::cout << calculatedSolutionValue << std::endl;
        }
    } catch (std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    return 0;
}
