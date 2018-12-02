#include <iostream>
#include "InputFileReader.h"
#include "QAP.h"

int main(int argc, char** argv) {

    try {
        InputFileReader reader{R"(C:\Users\Rafal\Desktop\magisterka\test instances\Bur\bur26a.dat)"};
        auto weights = reader.getWeights();
        auto distances = reader.getDistances();

        QAP qap{weights, distances};

        Permutation<unsigned> p{26, 15, 11, 7, 4, 12, 13, 2, 6, 18, 1, 5, 9, 21, 8, 14, 3, 20, 19, 25, 17, 10, 16, 24, 23, 22};
        for (auto& val : p)
            val -= 1;

        auto value = qap.getValue(p);
        std::cout << value << std::endl;
    } catch (std::exception& e) {
        std::cerr << e.what() << std::endl;
    }

    return 0;
}