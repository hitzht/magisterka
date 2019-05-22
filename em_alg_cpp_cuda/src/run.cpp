#include "run.h"
#include <chrono>
#include <iostream>

#include "cpu/CUDAUtils.h"
#include "cpu/InputFileReader.h"
#include "cpu/SolutionFileReader.h"
#include "cpu/QAP.h"
#include "cpu/PermutationFactory.h"
#include "cpu/AlgorithmInput.h"
#include "gpu/EMAlgorithm.h"

std::tuple<unsigned, unsigned, long> run(const std::string& inputPath, const std::string& solutionPath,
                                           unsigned blocks, unsigned populationPerBlock, unsigned iterations,
                                           double neighborhoodDistance) {
    CUDAUtils::checkParameters(blocks, populationPerBlock);

    std::cout << inputPath << std::endl;

    InputFileReader reader{inputPath};

    auto dimension = reader.getInstanceSize();
    auto weights = reader.getWeights();
    auto distances = reader.getDistances();

    SolutionFileReader solutionReader{solutionPath};

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
    algorithmInput.iterations = iterations;
    algorithmInput.neighborhoodDistance = dimension * neighborhoodDistance;
    algorithmInput.weights = std::move(weights);
    algorithmInput.distances = std::move(distances);
    algorithmInput.permutations = std::move(permutations);

    auto firstTimeout = std::chrono::high_resolution_clock::now();

    EMAlgorithm algorithm{};
    auto calculatedSolutionValue = algorithm.solve(algorithmInput);

    auto secondTimeout = std::chrono::high_resolution_clock::now();
    auto durationTime = std::chrono::duration_cast<std::chrono::milliseconds>(secondTimeout - firstTimeout).count();

    return std::make_tuple(solutionValue , calculatedSolutionValue, durationTime);
}