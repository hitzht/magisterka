import os
import re
from typing import List, Tuple
from em_discrete_perform import execute

tests_per_iteration = 10


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


def perform_tests(test_instances, iterations, permutations_count, distance_factor):


    print(iterations, permutations_count, distance_factor)

    file_name = str(iterations) + "_" + str(permutations_count) + "_" + str(distance_factor) + ".txt"

    output_file = open(file_name, "w")

    for instance in test_instances:
        instance_name = instance[0]
        input_file = instance[1]
        solution_file = instance[2]

        results = []

        for i in range(tests_per_iteration):
            result = execute(input_file, solution_file, permutations_count, iterations, distance_factor)
            results.append(result)

        best_found = min([x[0] for x in results])
        best_diff = min([x[1] for x in results])

        average_value = sum([x[0] for x in results]) / tests_per_iteration
        average_diff = sum([x[1] for x in results]) / tests_per_iteration
        average_time = sum([x[2] for x in results]) / tests_per_iteration

        print(instance_name, best_found, best_diff, round(average_value, 2), round(average_diff, 2),
              round(average_time, 2))

        line = "{} & {} & {} & {:.2f} & {:.2f} & {:.2f} \\\\ \n"
        line = line.format(instance_name, best_found, best_diff, average_value, average_diff, average_time)

        output_file.write(line)

    output_file.close()


if __name__ == '__main__':
    instances_folder = "../test_instances/"

    test_instances = get_test_instances(instances_folder)
    test_instances.sort(key=lambda x: x[0])

    test_instance_pattern = ".*"

    regex_pattern = re.compile(test_instance_pattern)

    test_instances = list(filter(lambda x: regex_pattern.match(x[0]), test_instances))
    print(len(test_instances))

    perform_tests(test_instances, 500, 100, 0.6)
    # perform_tests(test_instances, 1000, 100, 0.6)
    # perform_tests(test_instances, 5000, 100, 0.6)
    # perform_tests(test_instances, 10000, 100, 0.6)
