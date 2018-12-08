#include <iostream>
#include "InputFileReader.h"
#include "QAP.h"
#include "EMAlgorithm.h"

void displayPermutation(const Permutation<unsigned>& permutation) {
    for (const auto& elem : permutation)
        std::cout << elem << " ";

    std::cout << std::endl;
}

void displayUsage();

int main(int argc, char** argv) {
    try {
        InputFileReader reader{R"(C:\Users\Rafal\Desktop\magisterka\test instances\Bur\bur26a.dat)"};

        auto instanceSize = reader.getInstanceSize();
        auto weights = reader.getWeights();
        auto distances = reader.getDistances();

        QAP qap{weights, distances};
        EMAlgorithm algorithm{qap};

        auto solution = algorithm.solve(instanceSize, 200, 100000, 8);

        displayPermutation(solution);
        std::cout << qap.getValue(solution) << std::endl;
        std::cout << qap.getValue(solution) - 5426670 << std::endl;
        std::cout << double(qap.getValue(solution) - 5426670)/5426670 * 100 << std::endl;
    } catch (std::exception& e) {
        std::cerr << e.what() << std::endl;
    }

    return 0;
}

void displayUsage() {
    std::cout << "Usage: " << std::endl;
}
