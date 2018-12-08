#include "PMX.h"
#include <exception>
#include <algorithm>
#include <list>
#include <string>
#include <numeric>

Permutation<unsigned>
PMX::perform(const Permutation<unsigned> &first, const Permutation<unsigned> &second, unsigned start, unsigned end) {
    if (first.size() != second.size())
        throw std::invalid_argument{"Permutations have different sizes"};

    if (start >= first.size())
        throw std::invalid_argument{"Invalid start index"};

    if (end >= first.size())
        throw std::invalid_argument{"Invalid end index"};


    Permutation<unsigned> result(first.size(), std::numeric_limits<unsigned>::max());

    std::copy(first.begin() + start, first.begin() + end + 1, result.begin() + start);

    std::list<unsigned> tmp;

    for (unsigned i = start; i <= end; i++) {
        if (!isValueInPermutationSegment(second[i], first, start, end))
            tmp.push_back(second[i]);
    }

    for (unsigned value : tmp) {
        switchValue(value, value, first, second, result, start, end);
    }

    for (unsigned i = 0; i < result.size(); i++) {
        if (result[i] == std::numeric_limits<unsigned>::max())
            result[i] = second[i];
    }

    return result;
}

unsigned
PMX::getIndexOfValue(const Permutation<unsigned> &permutation, unsigned value) {
    for (unsigned i = 0; i < permutation.size(); i++) {
        if (permutation[i] == value)
            return i;
    }

    throw std::runtime_error{"Could not find value: " + std::to_string(value)};
}

bool
PMX::isValueInPermutationSegment(unsigned value, const Permutation<unsigned> &p, unsigned start, unsigned end) {
    return std::find(p.begin() + start, p.begin() + end + 1, value) != (p.begin() + end + 1);
}

void
PMX::switchValue(unsigned valueToSwitch, unsigned valueToWrite, const Permutation<unsigned> &firstParent, const Permutation<unsigned> &secondParent,
                 Permutation<unsigned> &child, unsigned start, unsigned end) {
    auto indexInSecondParent = getIndexOfValue(secondParent, valueToSwitch);
    auto valueFromFirstParent = firstParent[indexInSecondParent];
    indexInSecondParent = getIndexOfValue(secondParent, valueFromFirstParent);
    if (indexInSecondParent >= start && indexInSecondParent <= end) {
        switchValue(valueFromFirstParent, valueToWrite, firstParent, secondParent, child, start, end);
    } else {
        child[indexInSecondParent] = valueToWrite;
    }
}
