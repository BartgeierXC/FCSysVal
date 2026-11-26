"""Build all data-based figures and tables in results/.

Usage: python scripts/build_figures.py [--skip-convergence] [--skip-transient] [--parameter-set iteration1|iteration2|all]

Coverage and the DoE process figure do not depend on the model and go to results/.
The D_0 evaluation (run D0) and the transient runs T01 to T04 go to results/<parameter set>/.
Simulation results are read from FCS_SIMULATION_ROOT/<parameter set>/<run>/output/.
"""

import argparse
import sys
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.io.coverage import build_coverage_results
from src.io.doe_process_figure import plot_doe_process
from src.io.measurement_uncertainty import evaluate_uncertainty, load_budget, relative_residuals
from src.io.run_workflow import (
    PARAMETER_SETS,
    evaluate_stationary,
    evaluate_transient,
    load_run_config,
    published_result,
    stationary_details,
)
from src.io.runtime_paths import resolve_config_path, resolve_results_dir
from src.io.validation_figures import plot_stationary_comparison, plot_time_comparison

TRANSIENT_RUNS = ('T01', 'T02', 'T03', 'T04')


def build_d0(parameter_set: str, out: Path) -> None:
    """Write metrics, uncertainty, details and the comparison figure of D_0."""
    results = evaluate_stationary('D0', result_path=published_result('D0', parameter_set))
    results['metrics'].to_csv(out / 'D0_metrics.csv')
    group_config = load_run_config('D0').group_config
    conditions = pd.read_csv(resolve_config_path(group_config.conditions_file))
    table, summary = evaluate_uncertainty(
        results,
        conditions,
        group_config.trigger,
        load_budget(resolve_config_path('measurement_uncertainty.yml')),
    )
    table.to_csv(out / 'D0_uncertainty_conditions.csv', index=False)
    summary.to_csv(out / 'D0_uncertainty_summary.csv')
    relative_residuals(table).to_csv(out / 'D0_relative_residuals.csv')
    for name, details in stationary_details(results, conditions, group_config.trigger).items():
        details.to_csv(out / f'D0_{name}.csv')
    figure_dir = out / 'figures'
    figure_dir.mkdir(exist_ok=True)
    plot_stationary_comparison(
        results['simulation_conditions'], results['measurement_conditions'], figure_dir / 'fcs_d0_group.png'
    )


def build_transient(parameter_set: str, out: Path) -> None:
    """Write metrics and the comparison over time of each transient run."""
    figure_dir = out / 'figures'
    figure_dir.mkdir(exist_ok=True)
    for run_id in TRANSIENT_RUNS:
        results = evaluate_transient(run_id, result_path=published_result(run_id, parameter_set))
        results['metrics'].to_csv(out / f'{run_id}_metrics.csv')
        fig = plot_time_comparison(results['simulation'], results['measurement'])
        fig.savefig(figure_dir / f'{run_id}_time_comparison.png', dpi=200, bbox_inches='tight')
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-convergence', action='store_true', help='Skip the pool convergence check')
    parser.add_argument('--skip-transient', action='store_true', help='Skip the transient runs T01 to T04')
    parser.add_argument('--parameter-set', default='all', choices=[*PARAMETER_SETS, 'all'])
    args = parser.parse_args()

    results_dir = resolve_results_dir()
    figure_dir = results_dir / 'figures'
    figure_dir.mkdir(exist_ok=True)

    plot_doe_process(figure_dir / 'doe_constrained_sampling_process.png')
    print(build_coverage_results(results_dir, convergence=not args.skip_convergence))

    sets = PARAMETER_SETS if args.parameter_set == 'all' else (args.parameter_set,)
    for parameter_set in sets:
        out = results_dir / parameter_set
        out.mkdir(exist_ok=True)
        build_d0(parameter_set, out)
        if not args.skip_transient:
            build_transient(parameter_set, out)
        print(f'{parameter_set}: results written to {out}')


if __name__ == '__main__':
    main()
