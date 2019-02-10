import os
from typing import List, Tuple
from em_qap_test import solve
from timeit import default_timer as timer

def get_test_instances(path: str) -> List[Tuple[str, str, str]]:
    test_instances = []

    dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

    for dir in dirs:
        solutions_dir = os.path.join(path, dir)
        test_files = [f for f in os.listdir(solutions_dir) if os.path.isfile(os.path.join(solutions_dir, f))]
        test_files_names = list(map(lambda s: s.split(".dat")[0], filter(lambda s: s.endswith(".dat"), test_files)))

        for name in test_files_names:
            dat_file = os.path.abspath(os.path.join(solutions_dir, name + ".dat"))
            out_file = os.path.abspath(os.path.join(solutions_dir, name + ".out"))
            test_instances.append((name, dat_file, out_file))

    return test_instances


def make_test(test_instance):
    name, input_file, solution_file = test_instance
    points_count = 10
    iterations = 100
    upper_bound = 10
    lower_bound = 0

    start = timer()
    result = solve(input_file, solution_file, points_count, iterations, upper_bound, lower_bound, show_progress=False)
    end = timer()

    best_values, avg_values, optimal_value, optimal_permutation, best_value, best_permutation = result

    try:
        diff = round(((best_value - optimal_value) / optimal_value) * 100, 2)
    except ZeroDivisionError:
        diff = 0

    time_diff = round(end - start, 2)
    s = '{0: <12} {1: <12} {2: <12} {3: <12} {4: <8}'.format(name, optimal_value, best_value, str(diff) + "%", time_diff)
    print(s)


if __name__ == '__main__':
    test_instances_path = "../test_instances/"
    test_instances = get_test_instances(test_instances_path)

    s = '{0: <12} {1: <12} {2: <12} {3: <12} {4: <8}'.format("test case", "optimal", "best found", "diff[%]", "ex. time[s]")
    print(s)

    for ti in test_instances:
        make_test(ti)
