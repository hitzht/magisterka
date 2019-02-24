#ifndef QAP_CPU_HAMMINGDISTANCE_H
#define QAP_CPU_HAMMINGDISTANCE_H

#include "QAPDataTypes.h"

class HammingDistance {
public:
    static unsigned calculate(const Permutation<unsigned>& p1, const Permutation<unsigned>& p2);
};


#endif