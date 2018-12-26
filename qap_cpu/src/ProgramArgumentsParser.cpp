#include "ProgramArgumentsParser.h"

ProgramArgumentsParser::ProgramArgumentsParser(int argc, char **argv):
    options("qap_cpu", "Implementation of electromagnetic algorithm for quadratic assignment problem") {
    parseResult = std::make_unique<cxxopts::ParseResult>(options.parse(argc, argv));
}

void ProgramArgumentsParser::addOptions() {
    options.add_options()
            ("f,input_file", "Path to file which contains weights and distances matrix.",
             cxxopts::value<std::string>())
            ("s,solution_file", "[optional]Path to file which contains optimal solution.",
             cxxopts::value<std::string>())
            ("p,population", "Size of solutions population.",
             cxxopts::value<unsigned>()->default_value("100")->implicit_value("100"))
            ("i,iterations", "Number of interactions to perform.",
             cxxopts::value<unsigned>()->default_value("1000")->implicit_value("1000"))
            ("d,distance", "Value of solution's neighborhood distance.",
             cxxopts::value<unsigned>()->default_value("1000")->implicit_value("1000"));
}

std::string ProgramArgumentsParser::getInputFile() {
    return this->parseResult->operator[]("input_file").as<std::string>();
}

bool ProgramArgumentsParser::hasSolutionFile() {
    return this->parseResult->operator[]("solution_file").count() > 0;
}

std::string ProgramArgumentsParser::getSolutionFile() {
    return this->parseResult->operator[]("solution_file").as<std::string>();
}

unsigned ProgramArgumentsParser::getPopulationSize() {
    return this->parseResult->operator[]("population").as<unsigned>();
}

unsigned ProgramArgumentsParser::getIterationsCount() {
    return this->parseResult->operator[]("iterations").as<unsigned>();
}

unsigned ProgramArgumentsParser::getNeighborhoodDistance() {
    return this->parseResult->operator[]("distance").as<unsigned>();
}
