import sys
from common import parse_arguments
from em_qap_gpu import solve
import matplotlib.pyplot as plt


def plot_results(optimal_value, best_values, avg_values, title=""):

    fig, ax = plt.subplots()
    fig.set_size_inches(10, 10)

    x = [v for v in range(0, iterations)]

    plt.plot(x, best_values, 'g', label='Best value')
    plt.plot(x, avg_values, 'b', label='Average value')
    plt.title(title)

    if optimal_value is not None:
        optimal = [optimal_value] * iterations
        plt.plot(x, optimal, 'r', label='Optimal value')

    plt.ylabel('Value')
    plt.xlabel('Iteration')
    plt.legend()
    plt.savefig('result.png', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    input_file, solution_file, points_count, iterations, upper_bound, lower_bound = parse_arguments(sys.argv[1:])
    best_values, optimal_value = solve(input_file, solution_file, points_count, iterations, upper_bound, lower_bound)
