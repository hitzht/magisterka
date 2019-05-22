#ifndef DIANA_RUN_H
#define DIANA_RUN_H

#include <tuple>
#include <string>

/**
 * @return pair: [best solution value, best found solution]
 */
std::tuple<unsigned, unsigned, long> run(const std::string& inputPath, const std::string& solutionPath,
        unsigned blocks, unsigned populationPerBlock, unsigned iterations, double neighborhoodDistance);


#endif //DIANA_RUN_H
