#include <gtest/gtest.h>
#include "../src/HammingDistance.h"

TEST(HammingDistance, calculate_onArgumentsWithDifferentSize_throwsException) {
    Permutation<unsigned> p1{1, 3, 4, 5, 6};
    Permutation<unsigned> p2{8, 9, 10};

    ASSERT_THROW(HammingDistance::calculate(p1, p2), std::invalid_argument);
}

TEST(HammingDistance, calculate_onValidArguments_returnsValidValue) {
    Permutation<unsigned> p1{1, 3, 4, 5, 6};
    Permutation<unsigned> p2{8, 9, 10, 5, 1};

    ASSERT_EQ(HammingDistance::calculate(p1, p2), 4);
}