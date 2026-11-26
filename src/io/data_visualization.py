import matplotlib.pyplot as plt
import numpy as np


def plot_mult_over_one(df_list, x_col=None, scatter=True, legend=None, fontsize=14, data_size=50, figsize=None,
                       from_zero=True,
                       save_path=None, x_label=None):
    """
    Plots multiple columns of DataFrames of the same kind against a single x column.
    Every column except the specified x column gets an own plot.
    All plots are shown in one figure with subplots.

    Parameters:
    df (list): List of DataFrames containing the data to plot.
    x_col (str): Name of the column to use as the x-axis.
    scatter (bool, optional): If True, plots will be scatter plots. If False, line plots. Default is True.
    save_path (str, optional): Path to save the plots. If None, plots are shown interactively.

    Returns:
    None
    """

    y_cols = [col for col in df_list[0].columns if col != x_col]
    num_plots = len(y_cols)

    if figsize is None:
        figsize = (10, 5 * num_plots)
    fig, axes = plt.subplots(num_plots, 1, figsize=figsize, sharex=True)

    if legend is None:
        legend = [f'DF {i + 1}' for i in range(len(df_list))]

    # Every DataFrame has an own color
    # colors = plt.cm.viridis(np.linspace(0, 1, len(df_list)))
    for i, df in enumerate(df_list):
        df_y_cols = [c for c in df.columns if c != x_col]
        for plot_idx, col in enumerate(df_y_cols):
            ax = axes[plot_idx] if num_plots > 1 else axes
            if x_col is not None:
                if scatter:
                    ax.scatter(df[x_col], df[col], label=f'{legend[i]}', s=data_size)
                else:
                    ax.plot(df[x_col], df[col], label=f'{legend[i]}', linewidth=data_size)
                ax.set_xlabel(x_col, fontsize=fontsize)
            else:
                if scatter:
                    ax.scatter(df.index, df[col], label=f'{legend[i]}', s=data_size)
                else:
                    ax.plot(df.index, df[col], label=f'{legend[i]}', linewidth=data_size)
                if x_label is not None:
                    ax.set_xlabel(x_label, fontsize=fontsize)
            ax.set_ylabel(col, fontsize=fontsize)
            ax.legend(fontsize=fontsize)
            ax.tick_params(axis='both', which='major', labelsize=fontsize)
            if from_zero:
                ax.set_ylim(bottom=0)  # Set y-axis from zero to max value
            # Set ylim max to maximum value of the column only if current limit is below
            if ax.get_ylim()[1] < df[col].max() * 1.1:
                ax.set_ylim(top=df[col].max() * 1.1)
            if ax.get_ylim()[0] > df[col].min() * 0.9:
                ax.set_ylim(bottom=df[col].min() * 0.9)
            ax.grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
