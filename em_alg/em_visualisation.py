import argparse
import sys
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from em.em_solver import EMSolver
from benchmark_functions.factory import BenchmarkFactory

init_points = None
local_points = None

def parse_arguments(args):
    parser = argparse.ArgumentParser(description='Script displays visualisation of electromagnetism-like algorithm')
    parser.add_argument('-l', '--list', help='display list of available test functions', action="store_true")
    parser.add_argument('-f', '--function', help='name of tested function', required=False)
    parser.add_argument('-p', '--points', help='points count', type=int, default=10)
    parser.add_argument('-i', '--iterations', help='iterations count', type=int, default=20)

    results = parser.parse_args(args)

    if results.list is False and results.function is None:
        parser.error("You have to provide at least -l(--list) or -f(--function)")

    return results.list, results.function, results.points, results.iterations


def list_available_functions():
    factory = BenchmarkFactory()
    functions = factory.get_all_functions()
    functions_2d = dict((k, v) for k, v in functions.items() if v.get_dimension() == 2)

    for function_name in functions_2d.keys():
        print(function_name)


def display_animation(function_name: 'str', points_count: int, iterations_count: int):
    factory = BenchmarkFactory()
    test_function = factory.get_benchmark_function(function_name)

    lower_bound, upper_bound = test_function.get_bounds()
    dimension = test_function.get_dimension()
    local_optima = list(zip(*test_function.get_local_optima()))

    solver = EMSolver(points_count, dimension, lower_bound, upper_bound, test_function.get_function())

    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.set_size_inches(8, 8, forward=True)

    x = np.linspace(lower_bound[0], upper_bound[0], 1000)
    y = np.linspace(lower_bound[1], upper_bound[1], 1000)
    X, Y = np.meshgrid(x, y)
    Z = test_function.get_function()(X, Y)

    area_plot = ax



    def update(num):
        input("Press Enter to continue...")

        if num == 0:
            print("inicjalizacja")
            area_plot.clear()
            area_plot.set_title("Inicjalizacja")
            area_plot.axis([lower_bound[0], upper_bound[0], lower_bound[1], upper_bound[1]])
            CS = area_plot.contour(X, Y, Z, 20, zorder=1, cmap='brg')
            plt.clabel(CS, inline=1, fontsize=8)

            area_plot.plot(local_optima[0], local_optima[1], 'bD', zorder=2)

            init_points = solver._EMSolver__points.copy()

            points = list(zip(*init_points))
            area_plot.plot(points[0], points[1], 'go', zorder=2, markersize=10)
        elif num == 1:
            print("local search")
            area_plot.clear()
            area_plot.set_title("Optymalizacja lokalna")
            area_plot.axis([lower_bound[0], upper_bound[0], lower_bound[1], upper_bound[1]])
            CS = area_plot.contour(X, Y, Z, 20, zorder=1, cmap='brg')
            plt.clabel(CS, inline=1, fontsize=8)

            area_plot.plot(local_optima[0], local_optima[1], 'bD', zorder=2)

            init_points = solver._EMSolver__points.copy()

            tmp = []
            for p in init_points:
                tmp.append(p.copy())

            points = list(zip(*init_points))
            area_plot.plot(points[0], points[1], 'go', zorder=2, markersize=10)

            solver._EMSolver__local_search()

            local_points = solver._EMSolver__points.copy()

            points = list(zip(*local_points))
            area_plot.plot(points[0], points[1], 'ro', zorder=2, markersize=10)

            movement = [ (a[0] - b[0], a[1] - b[1]) for a, b in zip(local_points, tmp)]
            movement = list(zip(*movement))
            points = list(zip(*tmp))
            area_plot.quiver(points[0], points[1], movement[0], movement[1], scale_units='xy', units='xy', angles='xy',
                             scale=1, zorder=2)

        elif num == 2:
            print("calculate forces")

            area_plot.clear()
            area_plot.set_title("Obliczenie sił")
            area_plot.axis([lower_bound[0], upper_bound[0], lower_bound[1], upper_bound[1]])
            CS = area_plot.contour(X, Y, Z, 20, zorder=1, cmap='brg')
            plt.clabel(CS, inline=1, fontsize=8)

            area_plot.plot(local_optima[0], local_optima[1], 'bD', zorder=2)

            init_points = solver._EMSolver__points.copy()

            points = list(zip(*init_points))
            area_plot.plot(points[0], points[1], 'go', zorder=2, markersize=10)

            forces = solver._EMSolver__calculate_forces()
            solver._EMSolver__move(forces)

            forces = list(zip(*forces))
            area_plot.quiver(points[0], points[1], forces[0], forces[1], scale_units='xy', units='xy', angles='xy',
                             scale=1, zorder=2)

        elif num == 3:
            print("move")

            area_plot.clear()
            area_plot.set_title("Przesunięcie punktów")
            area_plot.axis([lower_bound[0], upper_bound[0], lower_bound[1], upper_bound[1]])
            CS = area_plot.contour(X, Y, Z, 20, zorder=1, cmap='brg')
            plt.clabel(CS, inline=1, fontsize=8)

            area_plot.plot(local_optima[0], local_optima[1], 'bD', zorder=2)

            init_points = solver._EMSolver__points.copy()

            points = list(zip(*init_points))
            area_plot.plot(points[0], points[1], 'go', zorder=2, markersize=10)




    anim = animation.FuncAnimation(fig, update, fargs=(), interval=1000, blit=False)
    plt.show()


if __name__ == '__main__':

    list_functions, function_name, points, iterations = parse_arguments(sys.argv[1:])

    if list_functions:
        list_available_functions()
    else:
        display_animation(function_name, points, iterations)
