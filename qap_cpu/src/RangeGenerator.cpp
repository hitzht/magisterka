#include "RangeGenerator.h"

RangeGenerator::RangeGenerator():
    randomDevice{},
    engine{randomDevice()} {}

std::pair<unsigned, unsigned> RangeGenerator::get(unsigned max) {
    std::uniform_int_distribution<unsigned> distribution(0, max - 1);
    auto firstNumber = distribution(this->engine);
    auto secondNumber = distribution(this->engine);

    return std::make_pair(std::min(firstNumber, secondNumber), std::max(firstNumber, secondNumber));
}
