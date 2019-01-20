from em import initialize, calculate_forces, move, find_best_point
from benchmark_functions.test_function import himmelblau


if __name__ == '__main__':

    lower_bound = [-5, -5]
    upper_bound = [5, 5]
    points_count = 10
    iterations = 100
    local_search_iterations = 5
    local_search_factor = 0.2

    points = initialize(points_count, lower_bound, upper_bound)
    index = find_best_point(points, himmelblau)
    print(points[index], himmelblau(points[index]))

    while iterations:
        iterations -= 1

        # points = local_search(local_search_iterations, local_search_factor, lower_bound, upper_bound, points, himmelblau)
        forces = calculate_forces(points, himmelblau)
        points = move(points, forces, lower_bound, upper_bound, himmelblau)

    index = find_best_point(points, himmelblau)
    print(points[index], himmelblau(points[index]))











