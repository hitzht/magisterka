#ifndef QAP_CPU_RANGEGENERATOR_H
#define QAP_CPU_RANGEGENERATOR_H

#include <random>
#include <utility>

class RangeGenerator {
private:
    std::random_device randomDevice;
    std::mt19937 engine;

public:
    RangeGenerator();

    /**
     * Returns sorted pair of numbers from range [0, max)
     */
    std::pair<unsigned, unsigned> get(unsigned max);
};


#endif