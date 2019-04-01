import sys
import matplotlib.pyplot as plt
from em_qap_test import parse_arguments, solve


def plot_results(optimal_value, best_values, avg_values, title=""):

    fig, ax = plt.subplots()
    fig.set_size_inches(10, 10)

    x = [v for v in range(0, len(best_values))]

    plt.plot(x, best_values, 'g', label='Best value')
    plt.plot(x, avg_values, 'b', label='Average value')
    plt.title(title)

    if optimal_value is not None:
        optimal = [optimal_value] * len(best_values)
        plt.plot(x, optimal, 'r', label='Optimal value')

    plt.ylabel('Value')
    plt.xlabel('Iteration')
    plt.legend()
    plt.savefig('result.png', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    input_file, solution_file, points_count, iterations, upper_bound, lower_bound = parse_arguments(sys.argv[1:])
    result = solve(input_file, solution_file, points_count, iterations, upper_bound, lower_bound, show_progress=True)
    best_values, avg_values, optimal_value, optimal_permutation, best_value, best_permutation = result

    name = [v for v in input_file.split("/")][-1]
    title = "{}, points: {}, iterations: {}, upper: {}, lower: {}"
    title = title.format(name, points_count, iterations, upper_bound, lower_bound)

    plot_results(optimal_value, best_values, avg_values, title)
