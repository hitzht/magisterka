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

    bool hasHelp();
    void displayHelp();
    std::string getInputFile();
    std::string getSolutionFile();
    unsigned getBlocks();
    unsigned getPopulationSize();
    unsigned getIterationsCount();
    double getNeighborhoodDistance();

private:
    void addOptions();
};


#endif //QAP_CPU_PROGRAMARGUMENTSPARSER_H
