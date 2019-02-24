#ifndef QAP_CPU_QAP_H
#define QAP_CPU_QAP_H

#include "QAPInterface.h"

class QAP: public QAPInterface {
private:
    Matrix<unsigned> weights;
    Matrix<unsigned> distances;

public:
    QAP(const Matrix<unsigned> &weights, const Matrix<unsigned> &distances);

    unsigned getValue(const Permutation<unsigned> &permutation) override;
};


#endif
