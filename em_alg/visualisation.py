from em_alg import EMSolver
from test_functions import himmelblau
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


if __name__ == '__main__':

    lower_bound = [-5, -5]
    upper_bound = [5, 5]
    points_count = 5
    max_iterations = 40
    dimension = 2

    local_optima = [[3.0, 2.0], [-2.805118, 3.131312], [-3.779310, -3.283186], [3.584428, -1.848126]]
    local_optima = list(zip(*local_optima))

    solver = EMSolver(points_count, dimension, lower_bound, upper_bound, himmelblau)

    fig, ax = plt.subplots(nrows=1, ncols=2)
    fig.set_size_inches(10, 10, forward=True)
    ax[0].margins(0)
    ax[1].margins(0)

    x = np.linspace(lower_bound[0], upper_bound[0], 1000)
    y = np.linspace(lower_bound[1], upper_bound[1], 1000)
    X, Y = np.meshgrid(x, y)
    Z = himmelblau(X, Y)

    area_plot = ax[0]
    values_plot = ax[1]

    iterations = []
    best_values = []
    average_values = []

    def update(num):
        if num < max_iterations:
            area_plot.clear()
            area_plot.axis([lower_bound[0], upper_bound[0], lower_bound[1], upper_bound[1]])
            area_plot.contour(X, Y, Z, 20, zorder=1)
            area_plot.plot(local_optima[0], local_optima[1], 'bD', zorder=2)

            points, forces, moved_points = solver.next_iteration()

            points = list(zip(*points))
            area_plot.plot(points[0], points[1], 'ro', zorder=2)

            forces = list(zip(*forces))
            area_plot.quiver(points[0], points[1], forces[0], forces[1], scale_units='xy', units='xy', angles='xy', scale=1, zorder=2)

            tmp = list(zip(*moved_points))
            area_plot.plot(tmp[0], tmp[1], 'gX', zorder=4)

            best_point = moved_points[solver.find_best_point(moved_points)]
            best_value = himmelblau(best_point)
            average_value = sum([himmelblau(point) for point in moved_points]) / len(moved_points)

            iterations.append(num)
            best_values.append(best_value)
            average_values.append(average_value)

            values_plot.clear()
            values_plot.axis([0, 40, 0, max(max(best_values), max(average_values))])
            values_plot.plot(iterations, best_values, 'g')
            values_plot.plot(iterations, average_values, 'b')


    anim = animation.FuncAnimation(fig, update, fargs=(), interval=500, blit=False)
    plt.show()


