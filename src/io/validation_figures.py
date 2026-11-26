"""Publication figures of the validation runs."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.io.figure_style import COLUMN_WIDTH, MM, PRINT_STYLE, label, save_figure

# The six QoI of the manuscript. The stack temperature T_S is compared with the measured coolant outlet temperature.
STATIONARY_COLUMNS = (
    'P.sys [kW]', 'eta_sys [%]', 'F.Si.Cmp.C [kg/h]', 'p.Si.C [bara]', 'RH.Si.C [%]', 'T.So.CL [°C]',
)
SIMULATED_SYMBOL = {'T.So.CL [°C]': 'T_S'}


TIME_COLUMNS = (
    'U.S.AveCell [V]', 'P.S.el [kW]', 'RH.Si.C [%]', 'T.Si.C [°C]',
    'T.Si.CL [°C]', 'T.So.CL [°C]', 'p.So.C [bara]', 'pD.S.C [mbar]',
)


def plot_signals(df: pd.DataFrame, columns: list[str] | None = None) -> plt.Figure:
    """Plot signals over time, one axis per signal."""
    columns = list(df.columns if columns is None else columns)
    n_rows = (len(columns) + 1) // 2
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 2.6 * n_rows), sharex=True)
    axes = np.ravel(axes)
    for ax, column in zip(axes, columns):
        ax.plot(df.index, df[column], linewidth=0.8, color='steelblue')
        ax.set_ylabel(column, fontsize=9)
        ax.grid(alpha=0.3)
    for ax in axes[len(columns):]:
        ax.set_visible(False)
    for ax in axes[max(len(columns) - 2, 0):len(columns)]:
        ax.set_xlabel('Time [s]')
        ax.xaxis.set_tick_params(labelbottom=True)
    fig.tight_layout()
    return fig


def plot_time_comparison(
    sim_df: pd.DataFrame,
    source_df: pd.DataFrame,
    skip_time: float = 0,
    columns: tuple[str, ...] = TIME_COLUMNS,
) -> plt.Figure:
    """Plot simulation and experiment over time, the shaded span is the skipped warm-up."""
    columns = [c for c in columns if c in sim_df and c in source_df]
    n_rows = (len(columns) + 1) // 2
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 3 * n_rows), sharex=True)
    axes = np.ravel(axes)
    for ax, column in zip(axes, columns):
        ax.plot(source_df.index, source_df[column], linewidth=0.8, color='black', label='Experiment')
        ax.plot(sim_df.index, sim_df[column], linewidth=0.8, color='tab:orange', label='Simulation')
        if skip_time:
            ax.axvspan(0, skip_time, color='grey', alpha=0.2, linewidth=0)
        ax.set_ylabel(column, fontsize=9)
        ax.grid(alpha=0.3)
    for ax in axes[len(columns):]:
        ax.set_visible(False)
    for ax in axes[max(len(columns) - 2, 0):len(columns)]:
        ax.set_xlabel('Time [s]')
        ax.xaxis.set_tick_params(labelbottom=True)
    axes[0].legend(loc='best', fontsize=9)
    fig.tight_layout()
    return fig


def plot_stationary_comparison(
    sim_df: pd.DataFrame,
    source_df: pd.DataFrame,
    path: Path | None = None,
    columns: tuple[str, ...] = STATIONARY_COLUMNS,
) -> plt.Figure:
    """Plot simulation over experiment of each QoI at column width, the 1:1 line marks agreement.

    With *path*, the figure is saved as PNG and PDF and closed. Signals of the other air subsystem are left out.
    """
    columns = [c for c in columns if c in sim_df and c in source_df]
    n_cols = 2
    n_rows = -(-len(columns) // n_cols)
    with plt.rc_context(PRINT_STYLE):
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(COLUMN_WIDTH, 42 * MM * n_rows), layout='constrained')
        axes = np.ravel(axes)
        for ax, column in zip(axes, columns):
            measured, simulated = source_df[column].to_numpy(), sim_df[column].to_numpy()
            low, high = min(measured.min(), simulated.min()), max(measured.max(), simulated.max())
            limits = [low - 0.05 * (high - low), high + 0.05 * (high - low)]
            ax.plot(limits, limits, 'r--', linewidth=0.8, zorder=1)
            ax.scatter(measured, simulated, s=6, alpha=0.7, color='steelblue', edgecolors='none', zorder=2)
            ax.set_xlabel(label(column, 'meas. '))
            ax.set_ylabel(label(SIMULATED_SYMBOL.get(column, column), 'sim. '))
            ax.set_xlim(limits)
            ax.set_ylim(limits)
            ax.set_aspect('equal', adjustable='box')
            ax.grid(alpha=0.3)
        for ax in axes[len(columns):]:
            ax.set_visible(False)
        if path is not None:
            save_figure(fig, path)
            plt.close(fig)
    return fig
