#include "QAP.h"

QAP::QAP(const Matrix<unsigned> & weights, const Matrix<unsigned> &distances):
    weights (weights),
    distances(distances) {}

unsigned QAP::getValue(const Permutation<unsigned>& permutation) {
    unsigned value{0};

    for (unsigned a = 0; a < permutation.size(); a++) {
        for (unsigned b = 0; b < permutation.size(); b++) {
            auto weight = this->weights.at(a).at(b);
            auto firstPoint = permutation.at(a);
            auto secondPoint = permutation.at(b);
            auto distance = this->distances.at(firstPoint).at(secondPoint);
            value += weight * distance;
        }
    }

    return value;
}