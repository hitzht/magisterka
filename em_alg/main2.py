from em_alg import EMSolver
from benchmark_functions.test_function import himmelblau


if __name__ == '__main__':

    lower_bound = [-5, -5]
    upper_bound = [5, 5]
    points_count = 10
    iterations = 20
    dimension = 2

    solver = EMSolver(points_count, dimension, lower_bound, upper_bound, himmelblau)

    while iterations:
        iterations -= 1

        points = solver.next_iteration()
        index = solver.find_best_point(points)
        print(points[index], himmelblau(points[index]))
