#include <numeric>
#include "EMAlgorithm.h"
#include "HammingDistance.h"

EMAlgorithm::EMAlgorithm(QAPInterface &qap):
    qap{qap},
    factory{} {}

Permutation<unsigned>
EMAlgorithm::solve(unsigned instanceSize, unsigned populationSize, unsigned iterations, unsigned distance) {
    auto population = this->factory.get(instanceSize, populationSize);
    std::vector<Permutation<unsigned>> nextPopulation(populationSize);

    while (iterations--) {
        for (unsigned i =0; i < populationSize; i++) {
            auto surroundings = this->getSolutionSurroundings(population, i, distance);
            auto newSolution = performInjection(population[i], surroundings);
            nextPopulation[i] = newSolution;
        }

        population = nextPopulation;
    }

    return this->findBest(population);
}

std::vector<Permutation<unsigned>>
EMAlgorithm::getSolutionSurroundings(const std::vector<Permutation<unsigned>> &population, unsigned solutionIndex,
                                     unsigned distance) {
    std::vector<Permutation<unsigned>> surroundings;

    auto currentSolution = population[solutionIndex];
    auto currentValue = this->qap.getValue(currentSolution);

    for (unsigned i = 0; i < population.size(); i++)
        if (i != solutionIndex)
            if (HammingDistance::calculate(currentSolution, population[i]) <= distance)
                if (this->qap.getValue(population[i]) < currentValue)
                    surroundings.push_back(population[i]);

    return surroundings;
}

Permutation<unsigned>
EMAlgorithm::performInjection(Permutation<unsigned> solution, const std::vector<Permutation<unsigned>> &neighborhood) {
    for (const auto& neighbor: neighborhood) {
        auto range = this->rangeGenerator.get(solution.size());
        solution = this->pmx.perform(solution, neighbor, range.first, range.second);
    }

    return solution;
}

Permutation<unsigned> EMAlgorithm::findBest(const std::vector<Permutation<unsigned>> &population) {
    auto bestSolution = population[0];
    auto betsValue = this->qap.getValue(bestSolution);

    for (unsigned i = 1; i < population.size(); i++) {
        auto value = this->qap.getValue(population[i]);
        if (value < betsValue) {
            betsValue = value;
            bestSolution = population[i];
        }
    }

    return bestSolution;
}
