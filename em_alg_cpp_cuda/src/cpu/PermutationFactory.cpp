#include <numeric>
#include <algorithm>
#include "PermutationFactory.h"

PermutationFactory::PermutationFactory():
        randomDevice{},
        generator(randomDevice()) {}

Permutation<unsigned> PermutationFactory::get(unsigned size) {
    std::vector<unsigned> v(size);
    std::iota (std::begin(v), std::end(v), 0);
    std::shuffle(v.begin(), v.end(), this->generator);

    return v;
}

std::vector<Permutation<unsigned>> PermutationFactory::get(unsigned size, unsigned count) {
    std::vector<Permutation<unsigned int>> result{};

    for (unsigned i = 0; i < count; i++)
        result.push_back(PermutationFactory::get(size));

    return result;
}