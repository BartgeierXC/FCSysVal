"""Generic two-dimensional example of the constrained sampling process."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from src.io.figure_style import MM, PRINT_STYLE, TEXT_WIDTH, save_figure
from src.utils.val_doe import cal_val_dist, generate_sobol, propose_new_val_experiments


@plt.rc_context()
def plot_doe_process(path: Path) -> None:
    """Save the sampling, selection and coverage steps of the example."""
    X1, X2, X3 = "x1", "x2", "x3"

    # Bounding hyperrectangle B and current-dependent limits:
    #   x1 : (lower, upper) bounds
    #   xj : table of support points [x1^(p), l_j^(p), u_j^(p)]
    bounds = {
        X1: (0.0, 1.0),
        X2: np.array([
            [0.00, 0.30, 0.90],
            [0.30, 0.20, 0.95],
            [0.65, 0.35, 0.80],
            [1.00, 0.40, 0.65],
        ]),
        X3: np.array([
            [0.00, 0.10, 0.90],
            [1.00, 0.10, 0.90],
        ]),
    }

    # Coupling constraints g_q(x) >= 0: x2 within a band around a linear function of x3.
    rules = [
        lambda d: d[X2] >= 0.8 * d[X3] + 0.05 - 0.15,
        lambda d: d[X2] <= 0.8 * d[X3] + 0.05 + 0.15,
    ]

    # Sampling and selection settings
    M_EXP, SEED = 11, 7        # N_S = 2**M_EXP Sobol' points, fixed scrambling seed
    N_V = 10                   # number of new experiments

    # plt.rcParams.update({
    #     "font.family": "serif",
    #     "font.size": 16,
    #     "axes.labelsize": 17,
    #     "axes.titlesize": 17,
    #     "xtick.labelsize": 15,
    #     "ytick.labelsize": 15,
    #     "legend.fontsize": 14,
    # })
    # Steps a) and b): Sobol' sampling with conditional rescaling, then rejection of coupling constraints.
    N_S = 2 ** M_EXP

    sampled = generate_sobol(bounds, n=N_S, seed=SEED)                # conditional rescaling only
    pool = generate_sobol(bounds, n=N_S, seed=SEED, rules=rules)      # + rejection, pool P

    keep = np.ones(len(sampled), dtype=bool)
    for rule in rules:
        keep &= np.asarray(rule(sampled), dtype=bool)
    rejected = sampled[~keep]

    # Farthest-first selection from the pool without prior dataset (seed: centroid-nearest point).
    D = propose_new_val_experiments(pool, n=N_V)
    cov = cal_val_dist(D, pool)

    x_grid = np.linspace(*bounds[X1], 400)
    t2 = bounds[X2]
    lo_grid = np.interp(x_grid, t2[:, 0], t2[:, 1])
    hi_grid = np.interp(x_grid, t2[:, 0], t2[:, 2])

    C_ENV, C_ENV_FILL = "#1f4e79", "#d9e8f5"
    C_POOL, C_NEW = "#2c5f8a", "#b85c38"

    plt.rcParams.update(PRINT_STYLE)

    fig = plt.figure(figsize=(TEXT_WIDTH, 72 * MM), layout="constrained")
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.04])
    axes = [fig.add_subplot(gs[0, 0])]
    axes += [fig.add_subplot(gs[0, k], sharex=axes[0], sharey=axes[0]) for k in (1, 2)]
    cax = fig.add_subplot(gs[0, 3])
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)


    def style_axis(ax, title, label):
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(0.0, 1.0)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(r"$x_1$")
        ax.set_title(title, pad=4)
        ax.text(0.86, 0.96, label, transform=ax.transAxes, va="top", ha="left", fontweight="bold")
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])


    def draw_bounds(ax, fill=True):
        # bounds of x1 (projection of B) and current-dependent limits of x2
        ax.axvline(bounds[X1][0], color="#666666", lw=0.7, ls="--", zorder=1)
        ax.axvline(bounds[X1][1], color="#666666", lw=0.7, ls="--", zorder=1)
        if fill:
            ax.fill_between(x_grid, lo_grid, hi_grid, color=C_ENV_FILL, alpha=0.6, lw=0, zorder=0)
        ax.plot(x_grid, lo_grid, color=C_ENV, lw=0.8, zorder=1)
        ax.plot(x_grid, hi_grid, color=C_ENV, lw=0.8, zorder=1)


    # (a) sampling with conditional rescaling, rejection -> pool
    ax = axes[0]
    draw_bounds(ax)
    ax.scatter(rejected[X1], rejected[X2], s=2, c="0.75", linewidths=0, zorder=2)
    ax.scatter(pool[X1], pool[X2], s=2, c=C_POOL, alpha=0.85, linewidths=0, zorder=3)
    ax.plot(t2[:, 0], t2[:, 1], "o", ms=2.5, color=C_ENV, zorder=4)
    ax.plot(t2[:, 0], t2[:, 2], "o", ms=2.5, color=C_ENV, zorder=4)
    ax.text(0.9, 0.06, rf"$N_{{\mathcal{{S}}}}={N_S}$" "\n" rf"$N_P={len(pool)}$",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7)
    ax.set_ylabel(r"$x_2$")
    style_axis(ax, "Sampling and filtering", "(a)")

    # (b) farthest-first selection, numbered in order of selection
    ax = axes[1]
    draw_bounds(ax, fill=False)
    ax.scatter(pool[X1], pool[X2], s=2, c="0.80", linewidths=0, zorder=2)
    ax.scatter(D[X1], D[X2], s=48, c=C_NEW, edgecolors="#3a1f14", linewidths=0.4, zorder=3)
    for i, (u, v) in enumerate(zip(D[X1], D[X2]), start=1):
        ax.text(u, v, str(i), ha="center", va="center", fontsize=5, color="white", zorder=5)
    style_axis(ax, "Farthest-first selection", "(b)")

    # (c) coverage map rho_j of the design on the pool
    ax = axes[2]
    draw_bounds(ax, fill=False)
    order = np.argsort(cov.r)
    sc = ax.scatter(pool[X1].to_numpy()[order], pool[X2].to_numpy()[order], c=cov.r[order],
                    s=2.5, cmap="viridis", linewidths=0, zorder=2, vmin=0.0, vmax=cov.d_max)
    ax.scatter(D[X1], D[X2], s=10, facecolors="none", edgecolors="k", linewidths=0.5, zorder=4)
    j_max = int(np.argmax(cov.r))
    ax.scatter(pool[X1].iloc[j_max], pool[X2].iloc[j_max], s=45, marker="*", c="#d62728",
               edgecolors="k", linewidths=0.4, zorder=5)
    ax.text(0.9, 0.06,
            rf"$\rho_{{\max}}={cov.d_max:.2f}$",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7)
    cb = fig.colorbar(sc, cax=cax)
    cb.set_label(r"$\rho_j$")
    style_axis(ax, r"Coverage of $\mathcal{D}$", "(c)")

    # one legend for the whole figure
    fig.legend(handles=[
        Line2D([], [], marker="o", ls="", ms=4, color=C_POOL, label=r"pool $\mathcal{P}$"),
        Line2D([], [], marker="o", ls="", ms=4, color="0.75", label=r"rejected ($g_q<0$)"),
        Line2D([], [], marker="o", ls="", ms=4, color=C_ENV, label=r"support points of $x_2^{-}, x_2^{+}$"),
        Line2D([], [], marker="o", ls="", ms=8, color=C_NEW, label=rf"selected, in order ($n_v={len(D)}$)"),
        Line2D([], [], marker="o", ls="", ms=5, mfc="none", mec="k", label=r"design $\mathcal{D}$"),
        Line2D([], [], marker="*", ls="", ms=9, mfc="#d62728", mec="k", label=r"defines $\rho_{\max}$"),
    ], loc="outside lower center", ncol=3, frameon=False, handletextpad=0.2, columnspacing=1.4)

    save_figure(fig, path)
    plt.close(fig)
