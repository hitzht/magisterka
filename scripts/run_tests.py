import argparse
import os
from subprocess import Popen, PIPE
from typing import List, Tuple


def run_program(path: str, test_instance: Tuple[str, str]):
    args = [path]

    input_file = "--input_file={}".format(test_instance[0])
    solution_file = "--solution_file={}".format(test_instance[1])

    args.append(input_file)
    args.append(solution_file)
    args.append("--population=200")
    args.append("--iterations=2000")
    args.append("--distance=6")

    process = Popen(args, stdout=PIPE)
    (output, err) = process.communicate()

    exit_code = process.wait()
    if exit_code == 0:
        print(os.path.split(test_instance[0])[1], output)
    else:
        print(os.path.split(test_instance[0])[1], err)


def list_directories(path: str):
    dirs = os.listdir(path)
    return dirs


def get_test_instances(path: str) -> List[Tuple[str, str]]:
    test_instances = []

    dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

    for dir in dirs:
        solutions_dir = os.path.join(path, dir)
        test_files = [f for f in os.listdir(solutions_dir) if os.path.isfile(os.path.join(solutions_dir, f))]
        test_files_names = list(map(lambda s: s.split(".dat")[0], filter(lambda s: s.endswith(".dat"), test_files)))

        for name in test_files_names:
            dat_file = os.path.abspath(os.path.join(solutions_dir, name + ".dat"))
            out_file = os.path.abspath(os.path.join(solutions_dir, name + ".out"))
            test_instances.append((dat_file, out_file))

    return test_instances


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run implementation of electromagnetic algorithm for quadratic '
                                                 'assignment problem on each test instance')

    parser.add_argument("-p", "--program_path", required=True, help="Path to program binary file")
    parser.add_argument("-t", "--test_instances_dir", required=True, help="Path to directory, where test instances "
                                                                          "are stored")

    args = parser.parse_args()
    print(args.program_path)
    print(args.test_instances_dir)

    test_instances = get_test_instances(args.test_instances_dir)

    for test_instance in test_instances:
        run_program(args.program_path, test_instance)
