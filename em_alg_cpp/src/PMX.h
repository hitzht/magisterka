#ifndef QAP_CPU_PMX_H
#define QAP_CPU_PMX_H

#include "QAPDataTypes.h"

/**
 * Performs
 */
class PMX {
public:
    Permutation<unsigned>
    perform(const Permutation<unsigned> &first, const Permutation<unsigned> &second, unsigned start, unsigned end);

private:
    bool
    isValueInPermutationSegment(unsigned value, const Permutation<unsigned> &permutation, unsigned start, unsigned end);

    unsigned
    getIndexOfValue(const Permutation<unsigned> &permutation, unsigned value);

    void
    switchValue(unsigned value, unsigned valueToWrite, const Permutation<unsigned>& firstParent, const Permutation<unsigned>& secondParent,
                Permutation<unsigned>& child, unsigned start, unsigned end);
};

#endif