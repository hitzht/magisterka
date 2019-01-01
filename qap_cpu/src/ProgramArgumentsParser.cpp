#include "ProgramArgumentsParser.h"

ProgramArgumentsParser::ProgramArgumentsParser(int argc, char **argv):
    options("qap_cpu", "Implementation of electromagnetic algorithm for quadratic assignment problem") {
    this->addOptions();
    parseResult = std::make_unique<cxxopts::ParseResult>(options.parse(argc, argv));
}

void ProgramArgumentsParser::addOptions() {
    options.add_options()
            ("h,help", "Print help")
            ("f,input_file", "Path to file which contains weights and distances matrix.",
             cxxopts::value<std::string>())
            ("s,solution_file", "[optional]Path to file which contains optimal solution.",
             cxxopts::value<std::string>())
            ("p,population", "Size of solutions population.",
             cxxopts::value<unsigned>()->default_value("100")->implicit_value("100"))
            ("i,iterations", "Number of interactions to perform.",
             cxxopts::value<unsigned>()->default_value("1000")->implicit_value("1000"))
            ("d,distance", "Value of solution's neighborhood distance.",
             cxxopts::value<unsigned>()->default_value("6")->implicit_value("6"));
}

bool ProgramArgumentsParser::hasHelp() {
    return this->parseResult->count("help") > 0;
}

void ProgramArgumentsParser::displayHelp() {
    std::cout << this->options.help({""}) << std::endl;
}

std::string ProgramArgumentsParser::getInputFile() {
    if (this->parseResult->count("input_file")) {
        return this->parseResult->operator[]("input_file").as<std::string>();
    } else {
      this->displayHelp();
      throw std::runtime_error("Missing argument: input_file");
    }
}

bool ProgramArgumentsParser::hasSolutionFile() {
    return this->parseResult->operator[]("solution_file").count() > 0;
}

std::string ProgramArgumentsParser::getSolutionFile() {
    if (this->parseResult->count("solution_file")) {
        return this->parseResult->operator[]("solution_file").as<std::string>();
    } else {
        this->displayHelp();
        throw std::runtime_error("Missing argument: solution_file");
    }
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
