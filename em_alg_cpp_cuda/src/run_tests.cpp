#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <limits>
#include <algorithm>
#include <numeric>
#include <iomanip>

#include "run.h"

const std::vector<std::string> testInstances{
//    "bur26a",
//    "bur26b",
//    "bur26c",
//    "bur26d",
//    "bur26e",
//    "bur26f",
//    "bur26g",
//    "bur26h",
//    "chr22a",
//    "chr22b",
//    "esc32a",
//    "esc32b",
//    "esc32c",
//    "esc32d",
//    "esc32e",
//    "esc32g",
//    "esc64a",
//    "lipa30a",
//    "lipa30b",
//    "lipa40a",
//    "lipa40b",
//    "lipa50a",
//    "lipa60a",
//    "sko42",
//    "sko49",
//    "sko56",
//    "ste36a",
//    "ste36b",
//    "tho40",
//    "wil50",

    "chr22a",
    "esc32a",
    "ste36a",
    "wil50"
};

const std::string testsDirectory{"/home/rafal/Workspace/magisterka/test_instances/"};

void runTests(unsigned blocks , unsigned permutations, unsigned iterations, double neighborhoodDistance,
        unsigned testsPerInstance) {
    std::string outputFileName = "output_" + std::to_string(blocks) + "_" + std::to_string(permutations)
            + "_" + std::to_string(iterations) + "_" + std::to_string(int(neighborhoodDistance * 100)) + ".txt";

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
        std::vector<long> calculationTimes;

        try {
            for (unsigned i = 0; i < testsPerInstance; i++) {
                auto [solutionValue, calculatedSolution, calculationTime] = run(inputFile, solutionFile, blocks,
                        permutations, iterations, neighborhoodDistance);
                originalSolution = solutionValue;
                values.push_back(calculatedSolution);
                calculationTimes.push_back(calculationTime);

                if (calculatedSolution < bestSolution)
                    bestSolution = calculatedSolution;
            }

            auto diffBest = originalSolution != 0 ? double(bestSolution)/double(originalSolution) * 100 - 100 : 0;
            auto average = static_cast<unsigned>(std::accumulate(values.begin(), values.end(), 0.0)/values.size());
            auto diffAverage = originalSolution != 0 ? double(average)/double(originalSolution) * 100 - 100 : 0;
            auto averageTime = static_cast<unsigned>(std::accumulate(calculationTimes.begin(),
                    calculationTimes.end(), 0.0)/calculationTimes.size());

            output << testInstance << " & "
                   //<< originalSolution << " & "
                   << bestSolution << " & " << std::setprecision(2)
                   << diffBest << " & "<< average << " & " << std::setprecision (2) << diffAverage << " & "
                   << unsigned(averageTime) << " \\\\ " << std::endl << "\\hline" << std::endl;
        } catch (std::exception& e) {
            std::cerr << "received exception: " << e.what() << std::endl;
        }
    }
}

int main() {
    unsigned blocks = 10;
    unsigned permutations = 200;
    unsigned iterations = 250;
    double neighborhoodDistance = 0.4;
    unsigned testsPerInstance = 10;

    std::vector<double> neighborhood{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1};

    for (auto value : neighborhood)
        runTests(blocks, permutations, iterations, value, testsPerInstance);

    return 0;
}