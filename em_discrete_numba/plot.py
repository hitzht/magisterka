import matplotlib.pyplot as plt


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
    plt.show()
