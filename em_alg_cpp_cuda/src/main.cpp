#include <iostream>
#include <iomanip>
#include "cpu/ProgramArgumentsParser.h"
#include "run.h"

int main(int argc, char** argv) {
    try {
        ProgramArgumentsParser arguments(argc, argv);

        if (arguments.hasHelp()) {
            arguments.displayHelp();
            return 0;
        }

        auto inputFile = arguments.getInputFile();
        auto solutionFile = arguments.getSolutionFile();
        auto blocks = arguments.getBlocks();
        auto populationPerBlock = arguments.getPopulationSize();
        auto iterations = arguments.getIterationsCount();
        auto neighborhood = arguments.getNeighborhoodDistance();

        auto [bestSolutionValue, bestFoundSolution, calculationTime] =  run(inputFile, solutionFile, blocks, populationPerBlock, iterations, neighborhood);
        auto diffBest = bestSolutionValue != 0 ? double(bestFoundSolution)/double(bestSolutionValue) * 100 - 100 : 0;

        std::cout << bestSolutionValue << " " << bestFoundSolution << " " << std::setprecision(2) << diffBest << std::endl;
    } catch (std::exception& e) {
        std::cerr << e.what() << std::endl;
    }

    return 0;
}
