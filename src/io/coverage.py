"""Coverage of the validation operating domain and its publication figures."""

import io
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors

import data.parameters.fcs_variable_naming as fcn
import src.utils.physical_equations as pe
from src.io import data_processing as dp
from src.io.figure_style import COLUMN_WIDTH, MM, PRINT_STYLE, TEXT_WIDTH, label, save_figure, symbol
from src.io.run_workflow import load_measurements, load_run_config
from src.utils.val_doe import (
    _prepare_bound_arrays,
    cal_val_dist,
    fltr_bounds,
    fltr_rules,
    generate_sobol,
    propose_new_val_experiments,
    propose_val_experiments,
)

N_CELLS = 275
POOL_EXPONENT, POOL_SEED = 16, 42
RULE_TOLERANCE = 0.01  # The test profile holds setpoints rounded to two decimals
BOUND_TOLERANCE = 2e-2
CONVERGENCE_EXPONENTS = (14, 15, 16, 17, 18, 19, 20)

# Cathode air flow is bounded as stoichiometry, the data holds the air-flow setpoint only.
BOUNDS = {
    fcn.I_A_set: (100, 440),
    fcn.p_Si_A_set_bara: np.array([[100, 1.4, 1.7], [440, 1.8, 2.4]]),
    fcn.C_Stoic: np.array([[100, 2.2, 2.3], [300, 1.8, 2.3], [440, 1.8, 2.3]]),
    fcn.p_Si_Ca_set_bara: np.array([[100, 1.2, 1.4], [440, 1.6, 2.2]]),
    fcn.v_ARB_req_rpm: np.array([[100, 3900, 5900], [300, 5900, 5900]]),
    fcn.SS_Delta_T_degC: np.array([[100, 2, 3], [440, 8, 12]]),
    fcn.SS_T_Stack_degC: np.array([[100, 70, 76], [440, 70, 72]]),
    fcn.DPT_Si_Ca_degC_set: np.array([[100, 46, 52]]),
}
SET_COLUMNS = list(BOUNDS)
MEASURED_COLUMNS = [
    fcn.I_A_act, fcn.p_Si_A_bara, fcn.mf_Cmp_kg_per_h, fcn.p_Si_Ca_bara,
    fcn.v_ARB_rpm, fcn.T_Si_CL_degC, fcn.T_So_CL_degC, fcn.DPT_Si_Ca_degC,
]
# Name and unit of each design variable in the table of operating conditions.
DESIGN_NAMES = {
    fcn.I_A_set: ('I_set', 'A'),
    fcn.p_Si_A_set_bara: ('p_A_set', 'bar'),
    fcn.C_Stoic: ('lambda_C_set', '-'),
    fcn.p_Si_Ca_set_bara: ('p_C_set', 'bar'),
    fcn.v_ARB_req_rpm: ('omega_ARB_set', 'rpm'),
    fcn.SS_Delta_T_degC: ('DeltaT_Cl_set', 'K'),
    fcn.SS_T_Stack_degC: ('T_Cl_set', '\N{DEGREE SIGN}C'),
    fcn.DPT_Si_Ca_degC_set: ('T_DP_C_set', '\N{DEGREE SIGN}C'),
}


def make_rules(tol: float = 0.0) -> list:
    """Return the coupling constraints, relaxed by *tol*."""
    return [
        lambda d: d[fcn.SS_T_Stack_degC] >= (d[fcn.DPT_Si_Ca_degC_set] * 1.1 + 9.91) - tol,
        lambda d: d[fcn.SS_T_Stack_degC] <= (d[fcn.DPT_Si_Ca_degC_set] * 1.17 + 17.8) + tol,
        lambda d: d[fcn.p_Si_A_set_bara] > d[fcn.p_Si_Ca_set_bara] + 0.15 - tol,
        lambda d: d[fcn.p_Si_A_set_bara] < d[fcn.p_Si_Ca_set_bara] + 0.3 + tol,
        lambda d: d[fcn.p_Si_A_set_bara] < d[fcn.C_Stoic] + 0.2 + tol,
        lambda d: d[fcn.SS_T_Stack_degC] + d[fcn.SS_Delta_T_degC] < 82 + tol,
        lambda d: d[fcn.p_Si_Ca_set_bara]
        < d[fcn.I_A_set] * d[fcn.C_Stoic] * 0.098278934 * 0.04701 + 0.5037 + tol,
    ]


def add_stoichiometry(df: pd.DataFrame) -> pd.DataFrame:
    """Add the cathode stoichiometry derived from the air-flow and current setpoints."""
    df = df.copy()
    df[fcn.C_Stoic] = pe.to_stoic(df[fcn.SS_vf_air_set_nl_per_min], df[fcn.I_A_set], N_CELLS)
    return df


def load_profile_steps() -> pd.DataFrame:
    """Return the step means of one run through the D_0 profile, the profile restarts afterwards."""
    config = load_run_config('COV')
    measured, _ = load_measurements(config)
    steps = dp.group_trigger_data(config, measured.astype(float))
    return steps[~steps[fcn.profile_step_no].duplicated()]


def operating_conditions(steps: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Merge the passes of each operating condition.

    Steps with identical setpoints are the passes of one condition. Returns the
    setpoints and the measured values of the conditions, and the step table.
    """
    merged, table = merge_passes(steps)
    return merged[SET_COLUMNS], merged[MEASURED_COLUMNS], table[['step', 'condition', 'pass']]


def merge_passes(steps: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the pass means indexed by the merge label, and the step table with label, condition and pass."""
    df = add_stoichiometry(steps)
    df = df[list(dict.fromkeys(SET_COLUMNS + MEASURED_COLUMNS + [fcn.profile_step_no]))]
    df = df[fltr_bounds(df, BOUNDS, BOUND_TOLERANCE)]
    df = df[fltr_rules(df, make_rules(RULE_TOLERANCE))]

    adjacency = squareform(pdist(df[SET_COLUMNS].to_numpy(dtype=float))) < 1e-6
    _, labels = connected_components(adjacency, directed=False)
    passes = pd.Series(labels).value_counts()
    if not (passes == 2).all():
        raise ValueError(f'Conditions without two passes: {passes[passes != 2].index.tolist()}')

    merged = df.groupby(labels, sort=True).mean().reset_index(drop=True)
    table = pd.DataFrame({'step': df[fcn.profile_step_no].astype(int).to_numpy(), 'label': labels})
    table = table.sort_values('step')
    numbers = {label: i + 1 for i, label in enumerate(dict.fromkeys(table['label']))}
    table['condition'] = table['label'].map(numbers)
    table['pass'] = table.groupby('condition').cumcount() + 1
    return merged, table


def conditions_table(steps: pd.DataFrame, study: 'CoverageStudy') -> pd.DataFrame:
    """Return the planned and realized operating conditions of D_0 and the proposed conditions of D_1.

    D_0: one row per condition with the steps of both passes, the setpoints (planned) and the
    measured values in the coordinates of the setpoints (realized), both averaged over the passes.
    D_1: the farthest-first additions to D_0 that reach the reference dispersion, not measured.
    """
    merged, table = merge_passes(steps)
    planned = merged[SET_COLUMNS]
    realized = to_setpoint_space(merged[MEASURED_COLUMNS], pd.Index(SET_COLUMNS))
    if not np.allclose(planned.to_numpy(dtype=float), study.design[SET_COLUMNS].to_numpy(dtype=float)):
        raise ValueError('The study design differs from the merged operating conditions')
    condition = table.drop_duplicates('label').set_index('label')['condition']
    passes = table.pivot(index='condition', columns='pass', values='step')

    def named(df: pd.DataFrame, kind: str) -> pd.DataFrame:
        return df.rename(columns={c: f'{name} {kind} [{unit}]' for c, (name, unit) in DESIGN_NAMES.items()})

    d0 = pd.DataFrame({'set': 'D0', 'condition': condition.reindex(merged.index).to_numpy(),
                       'status': 'measured'}, index=merged.index)
    d0['step pass 1'] = passes.loc[d0['condition'], 1].to_numpy()
    d0['step pass 2'] = passes.loc[d0['condition'], 2].to_numpy()
    d0 = pd.concat([d0, named(planned, 'planned'), named(realized, 'realized')], axis=1)
    d0 = d0.sort_values('condition')

    n0 = len(study.design)
    additions = study.augmented.iloc[n0:n0 + study.first_below][SET_COLUMNS].reset_index(drop=True)
    d1 = pd.DataFrame({'set': 'D1', 'condition': np.arange(n0 + 1, n0 + 1 + len(additions)),
                       'status': 'proposed, not measured'})
    d1 = pd.concat([d1, named(additions, 'planned')], axis=1)

    result = pd.concat([d0, d1], ignore_index=True)[d0.columns]
    for column in ('condition', 'step pass 1', 'step pass 2'):
        result[column] = result[column].astype('Int64')
    return result


def to_setpoint_space(measured: pd.DataFrame, columns: pd.Index) -> pd.DataFrame:
    """Express measured values in the coordinates of the setpoints."""
    result = measured.copy()
    air_flow = pe.from_mf_to_nl_air(measured[fcn.mf_Cmp_kg_per_h] / 3600.0)
    result[fcn.mf_Cmp_kg_per_h] = pe.to_stoic(air_flow, measured[fcn.I_A_act], N_CELLS)
    result[fcn.T_Si_CL_degC] = measured[fcn.T_So_CL_degC] - measured[fcn.T_Si_CL_degC]
    result[fcn.T_So_CL_degC] = measured[fcn.T_Si_CL_degC]
    result.columns = columns
    return result


@dataclass
class CoverageStudy:
    """Designs and distances behind the coverage figures."""

    design: pd.DataFrame
    measured: pd.DataFrame
    pool: pd.DataFrame
    distance: object
    reference: pd.DataFrame
    reference_distance: object
    sobol: tuple
    farthest_first: tuple
    augmented_n: list
    augmented_rho: list
    augmented: pd.DataFrame
    first_below: int


def sobol_statistics(pool: pd.DataFrame, n_exp_max: int = 10, n_rep: int = 200) -> tuple:
    """Return size, minimum, maximum and mean of rho_max of Sobol' designs after rejection."""
    rules = make_rules()
    stats = []
    for i in range(1, n_exp_max):
        lengths, d_max_values = [], []
        for rep in range(n_rep):
            df = generate_sobol(BOUNDS, n=2**i, seed=1000 * i + rep, rules=rules)
            if df.empty:
                continue
            lengths.append(len(df))
            d_max_values.append(cal_val_dist(df, pool, bounds=BOUNDS).d_max)
        if d_max_values:
            stats.append((np.mean(lengths), min(d_max_values), max(d_max_values), np.mean(d_max_values)))
    return tuple(map(np.array, zip(*stats)))


def run_study(design: pd.DataFrame, measured: pd.DataFrame) -> CoverageStudy:
    """Evaluate the design against the pool, the reference design and its augmentation."""
    pool = generate_sobol(BOUNDS, n=2**POOL_EXPONENT, seed=POOL_SEED, rules=make_rules())
    distance = cal_val_dist(design, pool, bounds=BOUNDS)
    sobol = sobol_statistics(pool)

    ff_n = list(range(2, int(max(sobol[0])) + 2, 2))
    ff_rho = [
        cal_val_dist(propose_new_val_experiments(pool, n=n, bounds=BOUNDS), pool, bounds=BOUNDS).d_max
        for n in ff_n
    ]
    reference = propose_new_val_experiments(pool, n=len(design), bounds=BOUNDS)
    reference_distance = cal_val_dist(reference, pool, bounds=BOUNDS)

    augmented = design.copy()
    seq_n, seq_rho = [len(design)], [distance.d_max]
    for _ in range(len(reference)):
        step = propose_val_experiments(pool, augmented, n=1, bounds=BOUNDS)
        augmented = pd.concat([augmented, step], ignore_index=True)
        seq_n.append(len(augmented))
        seq_rho.append(cal_val_dist(augmented, pool, bounds=BOUNDS).d_max)
    first_below = next(i for i, d in enumerate(seq_rho) if d < reference_distance.d_max)

    return CoverageStudy(
        design, measured, pool, distance, reference, reference_distance, sobol,
        (ff_n, ff_rho), seq_n, seq_rho, augmented, first_below,
    )


def pool_convergence(design: pd.DataFrame) -> pd.DataFrame:
    """Return rho_max of the reference design and of *design* on nested pools."""
    pools = {m: generate_sobol(BOUNDS, n=2**m, seed=POOL_SEED, rules=make_rules()) for m in CONVERGENCE_EXPONENTS}
    largest = pools[max(CONVERGENCE_EXPONENTS)]
    rows = []
    for m, pool in pools.items():
        reference = propose_new_val_experiments(pool, n=len(design), bounds=BOUNDS)
        rows.append({
            'N_Sobol': 2**m,
            'N_P': len(pool),
            'rho_max C (own pool)': cal_val_dist(reference, pool, bounds=BOUNDS).d_max,
            'rho_max C (largest pool)': cal_val_dist(reference, largest, bounds=BOUNDS).d_max,
            'rho_max D_0 (own pool)': cal_val_dist(design, pool, bounds=BOUNDS).d_max,
        })
    return pd.DataFrame(rows)


def summarize(study: CoverageStudy) -> dict:
    """Return the coverage numbers reported in the manuscript."""
    realized = cal_val_dist(
        to_setpoint_space(study.measured, study.pool.columns), study.pool, bounds=BOUNDS
    )
    n_valid, _, _, d_mean = study.sobol
    return {
        'n_0': len(study.design),
        'N_P': len(study.pool),
        'rho_max_planned': study.distance.d_max,
        'eta_planned': study.distance.eta,
        'rho_max_realized': realized.d_max,
        'eta_realized': realized.eta,
        'rho_max_reference': study.reference_distance.d_max,
        'sobol_mean': float(np.interp(len(study.design), n_valid, d_mean)),
        'k': study.first_below,
        'n_augmented': study.augmented_n[study.first_below],
        'rho_max_augmented': study.augmented_rho[study.first_below],
    }


def plot_sobol(study: CoverageStudy, path: Path) -> None:
    """Save rho_max of Sobol' and farthest-first designs over the design size, at column width."""
    n_valid, d_min, d_max, d_mean = study.sobol
    ff_n, ff_rho = study.farthest_first
    n_design = len(study.design)
    with plt.rc_context(PRINT_STYLE):
        fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 55 * MM), layout='constrained')
        ax.fill_between(n_valid, d_min, d_max, alpha=0.2, linewidth=0, label=r"Sobol$'$ min-max range")
        ax.plot(n_valid, d_mean, 'bo-', markersize=2, label=r"Sobol$'$ mean")
        ax.plot(ff_n, ff_rho, 'go-', markersize=2, label=r'Farthest-first on $\mathcal{P}$')
        ax.scatter(x=n_design, y=study.distance.d_max, color='red', s=40, marker='*',
                   label=r'$\mathcal{D}_0$', zorder=5)
        ax.axvline(x=n_design, color='red', linestyle='--', linewidth=0.8, label=rf'$n_v={n_design}$')
        ax.set_xlim(min(n_valid), max(n_valid))
        ax.set_xlabel(r'$n_v$')
        ax.set_ylabel(r'$\rho_{\max}$')
        ax.grid(alpha=0.3)
        ax.legend()
        save_figure(fig, path)
        plt.close(fig)


def plot_augmentation(study: CoverageStudy, path: Path) -> None:
    """Save rho_max over the size of the sequentially augmented design, at column width."""
    n_design = len(study.design)
    k = study.first_below
    with plt.rc_context(PRINT_STYLE):
        fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 55 * MM), layout='constrained')
        ax.plot(study.augmented_n, study.augmented_rho, 'o-', color='tab:green', markersize=2,
                label=r'$\mathcal{D}_0$ augmented by farthest-first')
        ax.scatter(n_design, study.distance.d_max, color='red', s=40, marker='*', zorder=5,
                   label=r'$\mathcal{D}_0$ ' + f'($n_v={n_design}$)')
        ax.axhline(study.reference_distance.d_max, color='black', linestyle=':', linewidth=1.2,
                   label=r'$\mathcal{C}$ ' + f'($n_v={n_design}$)')
        ax.scatter(study.augmented_n[k], study.augmented_rho[k], color='red', s=16, zorder=7,
                   label=rf'$\mathcal{{D}}_0\cup\mathcal{{D}}_1$ ($n_v={study.augmented_n[k]}$)')
        ax.set_xlabel(r'$n_v$')
        ax.set_ylabel(r'$\rho_{\max}$')
        ax.grid(alpha=0.3)
        ax.legend()
        save_figure(fig, path)
        plt.close(fig)


def largest_gaps(study: CoverageStudy, n_top: int = 5) -> pd.DataFrame:
    """Return rho_j and the normalized distance per setpoint of the farthest pool points."""
    design, pool, distance = study.design, study.pool, study.distance
    columns = list(BOUNDS)
    _, _, _, lower, upper, _ = _prepare_bound_arrays({c: BOUNDS[c] for c in columns})
    span = np.where(upper - lower == 0.0, 1.0, upper - lower)
    neighbors = NearestNeighbors(n_neighbors=1).fit(design[columns].to_numpy() / span)
    top = pool[columns].iloc[distance.r.argsort()[::-1][:n_top]].reset_index(drop=True)
    _, nearest = neighbors.kneighbors(top.to_numpy() / span)
    near = design[columns].iloc[nearest.ravel()].reset_index(drop=True)
    d_dim = np.abs(top.to_numpy() - near.to_numpy()) / span
    gaps = pd.DataFrame(d_dim, columns=[c.split(' [')[0] for c in columns])
    gaps.insert(0, 'rho', np.linalg.norm(d_dim, axis=1))
    gaps.index = pd.RangeIndex(1, n_top + 1, name='rank')
    return gaps


def plot_gap(study: CoverageStudy, path: Path, n_top: int = 5) -> None:
    """Save the projections of pool and designs and the distance of the farthest pool points, at column width."""
    design, pool, distance = study.design, study.pool, study.distance
    columns = list(BOUNDS)
    _, _, _, lower, upper, _ = _prepare_bound_arrays({c: BOUNDS[c] for c in columns})
    span = np.where(upper - lower == 0.0, 1.0, upper - lower)
    short = [c.split(' [')[0] for c in columns]

    neighbors = NearestNeighbors(n_neighbors=1).fit(design[columns].to_numpy() / span)
    gaps = largest_gaps(study, n_top)
    d_dim = gaps[short].to_numpy()
    rho = gaps['rho'].to_numpy()

    worst = pool[columns].iloc[np.flatnonzero(distance.r >= distance.d_max - 1e-12)]
    _, nearest = neighbors.kneighbors(worst.to_numpy() / span)
    nearest_design = design[columns].iloc[nearest.ravel()]
    additions = study.augmented.iloc[len(design):len(design) + study.first_below]

    with plt.rc_context(PRINT_STYLE):
        x_col = fcn.I_A_set
        y_cols = [c for c in columns if c != x_col]
        fig, axes = plt.subplots(4, 2, figsize=(COLUMN_WIDTH, 185 * MM), layout='constrained')
        for ax, y_col in zip(axes.ravel(), y_cols):
            ax.scatter(pool[x_col], pool[y_col], s=0.3, c='lightgray', rasterized=True, label=r'$\mathcal{P}$')
            ax.scatter(design[x_col], design[y_col], s=6, c='tab:blue', label=r'$\mathcal{D}_0$')
            ax.scatter(additions[x_col], additions[y_col], s=9, c='tab:green', marker='^', zorder=4,
                       label=r'$\mathcal{D}_1$')
            ax.scatter(worst[x_col], worst[y_col], s=30, c='red', marker='*', zorder=5,
                       label=r'point in $\mathcal{P}$ attaining $\rho_{\max}$')
            ax.scatter(nearest_design[x_col], nearest_design[y_col], s=16, facecolors='none',
                       edgecolors='red', linewidths=0.8, zorder=5, label=r'nearest point of $\mathcal{D}_0$')
            for (_, w), (_, n) in zip(worst.iterrows(), nearest_design.iterrows()):
                ax.plot([w[x_col], n[x_col]], [w[y_col], n[y_col]], 'r--', linewidth=0.6)
            ax.set_xlabel(label(x_col))
            ax.set_ylabel(label(y_col))
            ax.ticklabel_format(style='sci', axis='both', scilimits=(-3, 3), useMathText=True)
            ax.grid(alpha=0.3)
        handles, labels = axes.ravel()[0].get_legend_handles_labels()
        legend = fig.legend(handles, labels, loc='outside upper center', ncol=3, markerscale=1.5, columnspacing=1.0)

        ax = axes.ravel()[len(y_cols)]
        positions = np.arange(d_dim.shape[1])
        n_bars = d_dim.shape[0]
        width = 0.8 / n_bars
        for k in range(n_bars):
            ax.bar(positions + (k - (n_bars - 1) / 2) * width, d_dim[k], width=width,
                   label=rf'#{k + 1}: {rho[k]:.3f}')
        ax.set_xticks(positions)
        ax.set_xticklabels([symbol(c) for c in short], rotation=60, ha='right')
        ax.set_ylabel(r'$|\Delta\xi_j|$ (normalized on $\mathcal{B}$)')
        ax.grid(alpha=0.3, axis='y')
        ax.set_ylim(0, 1.8 * d_dim.max())
        # Header entry without handle in the upper left slot, legend filled column by column (2 x 3).
        bar_handles, bar_labels = ax.get_legend_handles_labels()
        header = Line2D([], [], linestyle='none', marker='none')
        ax.legend([header, *bar_handles], [r'$\rho_j$ rank', *bar_labels], fontsize=6, ncol=2, handlelength=0.8,
                  labelspacing=0.2, columnspacing=0.5, handletextpad=0.3, loc='upper left')
        # Center the figure legend on the gap between the two plot columns instead of the figure: fix the
        # constrained layout first, then move the legend horizontally.
        fig.savefig(io.BytesIO(), format='png', dpi=600)
        left, right = axes[0, 0].get_position(), axes[0, 1].get_position()
        top = legend.get_window_extent().transformed(fig.transFigure.inverted()).y1
        fig.set_layout_engine('none')
        legend.set_loc('upper center')
        legend.set_bbox_to_anchor(((left.x1 + right.x0) / 2, top), transform=fig.transFigure)
        save_figure(fig, path)
        plt.close(fig)


def build_coverage_results(results_dir: Path, convergence: bool = True) -> dict:
    """Write the coverage figures, numbers and tables to *results_dir*."""
    figure_dir = results_dir / 'figures'
    figure_dir.mkdir(parents=True, exist_ok=True)

    steps = load_profile_steps()
    design, measured, table = operating_conditions(steps)
    table.to_csv(results_dir / 'D0_operating_conditions.csv', index=False)
    study = run_study(design, measured)
    conditions_table(steps, study).to_csv(results_dir / 'operating_conditions.csv', index=False)
    plot_sobol(study, figure_dir / 'sobol_dist.png')
    plot_augmentation(study, figure_dir / 'cov_augment.png')
    plot_gap(study, figure_dir / 'cov_gap.png')
    largest_gaps(study).to_csv(results_dir / 'coverage_largest_gaps.csv')

    summary = summarize(study)
    (results_dir / 'coverage_numbers.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    if convergence:
        pool_convergence(design).to_csv(results_dir / 'pool_convergence.csv', index=False)
    return summary
