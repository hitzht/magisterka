#include <gtest/gtest.h>
#include "../src/PMX.h"

TEST(PMX, perform_onValidInputPermutations_returnsValidResult) {
    PMX pmx;
    Permutation<unsigned> p1{8, 4, 7, 3, 6, 2, 5, 1, 9, 0};
    Permutation<unsigned> p2{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

    auto p = pmx.perform(p1, p2, 3, 7);

    ASSERT_EQ(p.size(), 10);
    ASSERT_EQ(p.at(0), 0);
    ASSERT_EQ(p.at(1), 7);
    ASSERT_EQ(p.at(2), 4);
    ASSERT_EQ(p.at(3), 3);
    ASSERT_EQ(p.at(4), 6);
    ASSERT_EQ(p.at(5), 2);
    ASSERT_EQ(p.at(6), 5);
    ASSERT_EQ(p.at(7), 1);
    ASSERT_EQ(p.at(8), 8);
    ASSERT_EQ(p.at(9), 9);
}

TEST(PMX, perform_onValidInputPermutations_returnsValidResult2) {
    PMX pmx;
    Permutation<unsigned> p1{1, 2, 3, 4, 5, 6, 7, 8, 9};
    Permutation<unsigned> p2{4 ,5, 2, 1, 8, 7, 6, 9, 3};

    auto p = pmx.perform(p1, p2, 3, 6);

    ASSERT_EQ(p.size(), 9);
    ASSERT_EQ(p.at(0), 1);
    ASSERT_EQ(p.at(1), 8);
    ASSERT_EQ(p.at(2), 2);
    ASSERT_EQ(p.at(3), 4);
    ASSERT_EQ(p.at(4), 5);
    ASSERT_EQ(p.at(5), 6);
    ASSERT_EQ(p.at(6), 7);
    ASSERT_EQ(p.at(7), 9);
    ASSERT_EQ(p.at(8), 3);
}

TEST(PMX, perform_onValidInputPermutations_returnsValidResult3) {
    PMX pmx;
    Permutation<unsigned> p1{1, 5, 2, 8, 7, 4, 3, 6};
    Permutation<unsigned> p2{4, 2, 5, 8, 1, 3, 6, 7};

    auto p = pmx.perform(p1, p2, 2, 4);

    ASSERT_EQ(p.size(), 8);
    ASSERT_EQ(p.at(0), 4);
    ASSERT_EQ(p.at(1), 5);
    ASSERT_EQ(p.at(2), 2);
    ASSERT_EQ(p.at(3), 8);
    ASSERT_EQ(p.at(4), 7);
    ASSERT_EQ(p.at(5), 3);
    ASSERT_EQ(p.at(6), 6);
    ASSERT_EQ(p.at(7), 1);
}

