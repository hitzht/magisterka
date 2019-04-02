import argparse


def parse_arguments(args):
    def is_positive(value):
        if int(value) <= 0:
            raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)

        return int(value)

    parser = argparse.ArgumentParser(description='Script calculates qap solution by em discrete algorithm')
    parser.add_argument('-f', '--input_file', help='Path to file which contains weights and distances matrix.',
                        type=str, required=True)
    parser.add_argument('-s', '--solution_file', help='Path to file which contains optimal solution.', type=str,
                        required=False)
    parser.add_argument('-p', '--permutations', help='Number of permutations', type=is_positive, default=10)
    parser.add_argument('-i', '--iterations', help='Number of interactions to perform.', type=is_positive, default=100)
    parser.add_argument('-d', '--distance', help='Hamming distance between permutations', type=is_positive, default=6)

    res = parser.parse_args(args)

    return res.input_file, res.solution_file, res.permutations, res.iterations, res.distance
