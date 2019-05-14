#ifndef DIANA_RUN_H
#define DIANA_RUN_H

#include <utility>
#include <string>

/**
 * @return pair: [best solution value, best found solution]
 */
std::pair<unsigned, unsigned> run(const std::string& inputPath, const std::string& solutionPath,
        unsigned blocks, unsigned populationPerBlock, unsigned iterations, double neighborhoodDistance);


#endif //DIANA_RUN_H
