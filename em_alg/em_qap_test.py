import sys
import argparse
import numpy as np
from qap.input_file_reader import InputFileReader
from qap.solution_file_reader import SolutionFileReader
from qap.qap import QAP
from em.em_solver import EMSolver
from tqdm import tqdm


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


def solve(input_file, solution_file, points_count, iterations, upper_bound, lower_bound, show_progress=False):
    input_reader = InputFileReader(input_file)
    dimension, weights, distances = input_reader.read()

    qap = QAP(weights, distances)

    optimal_value = None
    optimal_permutation = None

    if solution_file is not None:
        solution_reader = SolutionFileReader(solution_file)
        solution_dimension, solution_value, solution_permutation = solution_reader.read()

        if solution_dimension != dimension:
            raise RuntimeError("Solution dimension is different than input dimension")

        if qap.get_value(solution_permutation) != solution_value:
            raise RuntimeError("Solution value does not match calculated solution permutation value")

        optimal_value = solution_value
        optimal_permutation = solution_permutation

    lower_bound = [-lower_bound] * dimension
    upper_bound = [upper_bound] * dimension
    tested_function = qap.get_em_function()

    solver = EMSolver(points_count, dimension, lower_bound, upper_bound, tested_function)

    best_values = []
    avg_values = []

    best_permutation = None
    best_value = 10 ** 10

    r = tqdm(range(iterations)) if show_progress else range(iterations)

    for it in r:
        _, _, points = solver.next_iteration()
        best_point_index = solver.find_best_point(points)

        iter_avg = sum((tested_function(p) for p in points)) / points_count
        iter_best_val = tested_function(points[best_point_index])

        if iter_best_val < best_value:
            best_value = iter_best_val
            best_permutation = points[best_point_index]

        best_values.append(iter_best_val)
        avg_values.append(iter_avg)

    return best_values, avg_values, optimal_value, optimal_permutation, best_value, np.argsort(best_permutation)


if __name__ == '__main__':
    input_file, solution_file, points_count, iterations, upper_bound, lower_bound = parse_arguments(sys.argv[1:])
    result = solve(input_file, solution_file, points_count, iterations, upper_bound, lower_bound, show_progress=True)
    best_values, avg_values, optimal_value, optimal_permutation, best_value, best_permutation = result

    try:
        diff = round(((best_value - optimal_value) / optimal_value) * 100, 2)
    except ZeroDivisionError:
        diff = 0

    print(optimal_permutation, optimal_value, best_permutation, best_value, diff)
