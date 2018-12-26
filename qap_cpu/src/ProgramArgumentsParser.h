#ifndef QAP_CPU_PROGRAMARGUMENTSPARSER_H
#define QAP_CPU_PROGRAMARGUMENTSPARSER_H

#include <memory>
#include <cxxopts.hpp>

class ProgramArgumentsParser {
private:
    cxxopts::Options options;
    std::unique_ptr<cxxopts::ParseResult> parseResult;

public:
    ProgramArgumentsParser(int argc, char** argv);

    std::string getInputFile();
    bool hasSolutionFile();
    std::string getSolutionFile();
    unsigned getPopulationSize();
    unsigned getIterationsCount();
    unsigned getNeighborhoodDistance();

private:
    void addOptions();
};


#endif //QAP_CPU_PROGRAMARGUMENTSPARSER_H
