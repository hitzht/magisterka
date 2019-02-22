import argparse


def parse_arguments(args):
    def is_positive(value):
        if int(value) <= 0:
            raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)

        return int(value)

    parser = argparse.ArgumentParser(description='Script displays visualisation of electromagnetism-like algorithm')
    parser.add_argument('-f', '--input_file', help='Path to file which contains weights and distances matrix.',
                        type=str, required=True)
    parser.add_argument('-s', '--solution_file', help='Path to file which contains optimal solution.', type=str,
                        required=False)
    parser.add_argument('-p', '--points', help='Number of points', type=is_positive, default=10)
    parser.add_argument('-i', '--iterations', help='Number of interactions to perform.', type=is_positive, default=10)
    parser.add_argument('-l', '--lower_bound', help='Lower bound value', type=int, default=0)
    parser.add_argument('-u', '--upper_bound', help='Upper bound value', type=int, default=10)

    res = parser.parse_args(args)

    if res.upper_bound < res.lower_bound:
        parser.error("Upper bound must be greater than lower bound")

    return res.input_file, res.solution_file, res.points, res.iterations, res.upper_bound, res.lower_bound
