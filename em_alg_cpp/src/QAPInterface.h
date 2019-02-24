#ifndef QAP_CPU_QAPINTERFACE_H
#define QAP_CPU_QAPINTERFACE_H

#include "QAPDataTypes.h"

class QAPInterface {
public:
    virtual unsigned getValue(const Permutation<unsigned> &permutation) = 0;
};

#endif
