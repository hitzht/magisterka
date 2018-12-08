#include <exception>
#include "HammingDistance.h"

unsigned HammingDistance::calculate(const Permutation<unsigned> &p1, const Permutation<unsigned> &p2) {
    if (p1.size() != p2.size())
        throw std::invalid_argument{"Permutation have different size"};

    unsigned differences{0};

    for (unsigned i = 0; i < p1.size(); i++)
        if (p1[i] != p2[i])
            differences++;

    return differences;
}
