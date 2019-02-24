#ifndef QAP_CPU_EMALGORITHM_H
#define QAP_CPU_EMALGORITHM_H

#include "QAPInterface.h"
#include "PermutationFactory.h"
#include "RangeGenerator.h"
#include "PMX.h"

class EMAlgorithm {
private:
    QAPInterface& qap;
    PermutationFactory factory;
    RangeGenerator rangeGenerator;
    PMX pmx;

public:
    explicit EMAlgorithm(QAPInterface &qap);

    Permutation<unsigned>
    solve(unsigned instanceSize, unsigned populationSize = 100, unsigned iterations = 1000, unsigned distance = 10);

private:
    std::vector<Permutation<unsigned>>
    getSolutionSurroundings(const std::vector<Permutation<unsigned>> &population, unsigned solutionIndex,
                            unsigned distance);

    Permutation<unsigned>
    performInjection(Permutation<unsigned> solution, const std::vector<Permutation<unsigned>>& population);

    Permutation<unsigned>
    findBest(const std::vector<Permutation<unsigned>> &population);
};


#endif
