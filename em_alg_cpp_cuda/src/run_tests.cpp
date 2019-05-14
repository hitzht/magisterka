#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <limits>
#include <algorithm>
#include <numeric>

#include "run.h"

const std::vector<std::string> testInstances{
    "bur26a",
    "bur26b",
    "bur26c",
    "bur26d",
    "bur26e",
    "bur26f",
    "bur26g",
    "bur26h",
    "chr22a",
    "chr22b",
    "esc32a",
    "esc32b",
    "esc32c",
    "esc32d",
    "esc32e",
    "esc32g",
    "esc32h",
    "esc64a",
    "lipa30a",
    "lipa30b",
    "lipa40a",
    "lipa40b",
    "lipa50a",
    "lipa60a",
    "sko42",
    "sko49",
    "sko56",
    "ste36a",
    "ste36b",
    "ste36c",
    "tho40",
    "wil50"
};

const std::string testsDirectory{"/home/rafal/Workspace/magisterka/test_instances/"};

void runTests(unsigned blocks , unsigned permutations, unsigned iterations, double neighborhoodDistance,
        unsigned testsPerInstance) {
    std::string outputFileName = "output_" + std::to_string(blocks) + "_" + std::to_string(permutations)
            + "_" + std::to_string(iterations) + "_" + std::to_string(neighborhoodDistance) + ".txt";

    std::ofstream output{outputFileName};

    if (!output.is_open()) {
        std::cerr << "could not open output file " << outputFileName << std::endl;
    }


    for (const auto& testInstance : testInstances) {
        auto inputFile = testsDirectory + testInstance + ".dat";
        auto solutionFile = testsDirectory + testInstance + ".sln";

        unsigned bestSolution = std::numeric_limits<unsigned>::max();
        unsigned originalSolution{0};
        std::vector<unsigned> values;

        try {
            for (unsigned i = 0; i < testsPerInstance; i++) {
                auto [solutionValue, calculatedSolution] = run(inputFile, solutionFile, blocks, permutations, iterations, neighborhoodDistance);
                originalSolution = solutionValue;
                values.push_back(calculatedSolution);

                if (calculatedSolution < bestSolution)
                    bestSolution = calculatedSolution;
            }

            auto diffBest = originalSolution != 0 ? double(bestSolution)/double(originalSolution) * 100 - 100 : 0;
            auto average = static_cast<unsigned>(std::accumulate(values.begin(), values.end(), 0.0)/values.size());
            auto diffAverage = originalSolution != 0 ? double(average)/double(originalSolution) * 100 - 100 : 0;

            output << testInstance << " " << originalSolution << " " << bestSolution << " " << diffBest << " "
                   << average << " " << diffAverage << std::endl;
        } catch (std::exception& e) {
            std::cerr << "received exception: " << e.what() << std::endl;
        }
    }
}

int main() {
    unsigned blocks = 10;
    unsigned permutations = 1000;
    unsigned iterations = 10;
    double neighborhoodDistance = 0.75;
    unsigned testsPerInstance = 10;

    runTests(blocks, permutations, iterations, neighborhoodDistance, testsPerInstance);


    return 0;
}