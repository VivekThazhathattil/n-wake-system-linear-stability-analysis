import matplotlib.pyplot as plt

def line_plotter(x, y, xlabel='X-axis', ylabel='Y-axis', title='Line Plot', legend=None, grid=True, save_path=None):
    """
    Plots a line plot with the given x and y data.

    Parameters:
    x (list or array): Data for the x-axis.
    y (list or array): Data for the y-axis.
    xlabel (str): Label for the x-axis. Default is 'X-axis'.
    ylabel (str): Label for the y-axis. Default is 'Y-axis'.
    title (str): Title of the plot. Default is 'Line Plot'.
    legend (list): List of legend labels. Default is None.
    grid (bool): Whether to show grid lines. Default is True.
    save_path (str): Path to save the plot image. Default is None (does not save).
    """
    plt.ion()
    fig, ax = plt.subplots()
    ax.plot(x, y, label=legend, color='blue')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if legend:
        ax.legend()

    if grid:
        ax.grid()

    if save_path:
        plt.savefig(save_path)

    return plt