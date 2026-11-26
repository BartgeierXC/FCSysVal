"""Measurement uncertainty of the quasi-stationary comparison pairs.

Combines the Type B uncertainty of the measurement chain with the Type A
uncertainty of the window means and marks the significant residuals.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import data.parameters.fcs_variable_naming as fcn
from src.utils.constants import U_LHV

N_CELLS = 275  # number of cells of the stack

COVERAGE_K = 2
QUANTITIES = (
    fcn.P_sys_kW, fcn.eta_sys_pct, fcn.U_cell_V, fcn.P_el_kW,
    fcn.P_CMP_kW, fcn.P_ARB_kW, fcn.P_CWP_W,
    fcn.mf_Cmp_kg_per_h, fcn.p_Si_Ca_bara, fcn.rh_Si_Ca_pct, fcn.T_So_CL_degC,
)
# Quantities judged by the MAE in their own unit instead of the MRE in %, with the threshold.
# The relative error of a temperature depends on the zero point of the scale.
ABSOLUTE_THRESHOLDS = {fcn.T_So_CL_degC: 2.0}  # K


def load_budget(path: Path) -> dict:
    """Return the uncertainty budget of a YAML file."""
    with open(path, encoding='utf8') as file:
        return yaml.safe_load(file)


def readings(values: pd.DataFrame, fixed: dict) -> dict[str, pd.Series | float]:
    """Return the instrument readings that full-scale limits refer to.

    The supplies of compressor and ARB run at a fixed voltage, so their current follows from the power.
    """
    return {
        'cmp_current': values[fcn.P_CMP_kW] * 1e3 / fixed['cmp_voltage'],
        'cmp_voltage': fixed['cmp_voltage'],
        'arb_current': values[fcn.P_ARB_kW] * 1e3 / fixed['arb_voltage'],
        'arb_voltage': fixed['arb_voltage'],
    }


def relative_type_b(budget: dict, values: pd.DataFrame | None = None) -> dict[str, float] | pd.DataFrame:
    """Return the relative standard uncertainty in % of each directly measured signal.

    `limits` and `chain` are relative to the reading. A `full_scale` entry gives a limit in % of a
    full-scale or nominal value `of`, which is converted to the reading of each operating
    condition. An `absolute` entry gives a limit in the unit of the signal itself, a `linear`
    entry one limit of an absolute part plus a percentage of the reading. These need
    *values*. Without them the result is one value per signal.
    """
    chain = budget.get('chain', {})
    uses_full_scale = any(entry.get(key) for entry in budget['signals'].values() for key in ('full_scale', 'absolute', 'linear'))
    if uses_full_scale and values is None:
        raise ValueError('full-scale, absolute and linear limits need the measured values')
    needs_readings = any(entry.get('full_scale') for entry in budget['signals'].values())
    reading = readings(values, budget['fixed']) if needs_readings else {}
    result = {}
    for signal, entry in budget['signals'].items():
        limits = list(entry.get('limits', [])) + [chain[name] for name in entry.get('chain', [])]
        squares = sum((a / np.sqrt(3)) ** 2 for a in limits)
        for item in entry.get('full_scale', []):
            squares = squares + (item['percent'] * item['of'] / reading[item['reading']] / np.sqrt(3)) ** 2
        for a in entry.get('absolute', []):
            squares = squares + (100 * a / values[signal].abs() / np.sqrt(3)) ** 2
        for item in entry.get('linear', []):
            limit = 100 * item['absolute'] / values[signal].abs() + item['percent']
            squares = squares + (limit / np.sqrt(3)) ** 2
        result[signal] = np.sqrt(squares)
    if not uses_full_scale:
        return {signal: float(v) for signal, v in result.items()}
    return pd.DataFrame({signal: pd.Series(v, index=values.index) if np.ndim(v) == 0 else v
                         for signal, v in result.items()})


def absolute_type_b(values: pd.DataFrame, relative: dict[str, float] | pd.DataFrame) -> pd.DataFrame:
    """Return the Type B standard uncertainty of each quantity in its own unit.

    The contributions of the measured signals are independent. The stack power is
    P_FC = N I U_c from the cell voltage U_c and the stack current I, the system power is the
    stack power minus the auxiliary powers. The system efficiency
    eta = P_sys / (N I U_LHV) = U_c / U_LHV - P_aux / (N I U_LHV) contains I in numerator and
    denominator, so it is propagated in the second form, with U_c, the auxiliary powers and I.
    """
    u = pd.DataFrame(index=values.index)
    for signal, percent in relative.items():
        u[signal] = values[signal].abs() * percent / 100
    # The pump power is measured in W, all other powers in kW.
    u_aux = [u[fcn.P_CMP_kW], u[fcn.P_ARB_kW], u[fcn.P_CWP_W] / 1000]
    u_cell, u_current = u[fcn.U_cell_V], u[fcn.I_A_act]
    current = values[fcn.I_A_act]
    u[fcn.P_el_kW] = values[fcn.P_el_kW].abs() * np.sqrt(
        (u_cell / values[fcn.U_cell_V]) ** 2 + (u_current / current) ** 2
    )
    u[fcn.P_sys_kW] = np.sqrt(u[fcn.P_el_kW] ** 2 + sum(x ** 2 for x in u_aux))
    p_aux = values[fcn.P_CMP_kW] + values[fcn.P_ARB_kW] + values[fcn.P_CWP_W] / 1000
    lhv_power = N_CELLS * current * U_LHV / 1e3  # kW
    u[fcn.eta_sys_pct] = 100 * np.sqrt(
        (u_cell / U_LHV) ** 2
        + sum((x / lhv_power) ** 2 for x in u_aux)
        + (p_aux * u_current / (lhv_power * current)) ** 2
    )
    return u


def type_a_of_conditions(
    u_a: pd.DataFrame, steps: pd.Series, conditions: pd.DataFrame
) -> pd.DataFrame:
    """Return the Type A uncertainty of the mean over the passes of each condition.

    *u_a* holds the Type A uncertainty of each window and *steps* its profile step.
    """
    step_to_condition = conditions.set_index('step')['condition']
    keep = steps.round().astype(int).isin(step_to_condition.index).to_numpy()
    labels = steps[keep].round().astype(int).map(step_to_condition).to_numpy()
    grouped = (u_a[keep] ** 2).groupby(labels)
    return (np.sqrt(grouped.sum()) / grouped.count()).rename_axis('condition')


def comparison_table(
    measured: pd.DataFrame, simulated: pd.DataFrame, u_b: pd.DataFrame, u_a: pd.DataFrame
) -> pd.DataFrame:
    """Return residuals, uncertainties and significance per condition and quantity."""
    frames = []
    for quantity in QUANTITIES:
        if quantity not in measured or quantity not in simulated:
            continue
        frame = pd.DataFrame({
            'quantity': quantity,
            'measured': measured[quantity],
            'simulated': simulated[quantity],
            'residual': measured[quantity] - simulated[quantity],
            'u_B': u_b[quantity],
            'u_A': u_a[quantity],
        })
        frame['u_D'] = np.sqrt(frame['u_B'] ** 2 + frame['u_A'] ** 2)
        frame['significant'] = frame['residual'].abs() > COVERAGE_K * frame['u_D']
        frames.append(frame)
    return pd.concat(frames).reset_index()


def summarize(table: pd.DataFrame, pass_difference: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Return the decidability, significance and repeatability of each quantity.

    *threshold* is the accuracy threshold in % of the MRE. Quantities in ABSOLUTE_THRESHOLDS are
    decided on the mean absolute u_D against their MAE threshold instead.
    """
    rows = {}
    for quantity, group in table.groupby('quantity', sort=False):
        relative = 100 * group['u_D'] / group['measured'].abs()
        difference = pass_difference[quantity].reindex(group['condition'])
        relative_difference = 100 * difference.abs().to_numpy() / group['measured'].abs().to_numpy()
        if quantity in ABSOLUTE_THRESHOLDS:
            decidable = COVERAGE_K * group['u_D'].mean() <= ABSOLUTE_THRESHOLDS[quantity]
        else:
            decidable = COVERAGE_K * relative.mean() <= threshold
        rows[quantity] = {
            'n': len(group),
            'mean relative u_B (%)': (100 * group['u_B'] / group['measured'].abs()).mean(),
            'mean relative u_A (%)': (100 * group['u_A'] / group['measured'].abs()).mean(),
            'mean relative u_D (%)': relative.mean(),
            'k u_D (%)': COVERAGE_K * relative.mean(),
            'mean u_D': group['u_D'].mean(),
            'k u_D': COVERAGE_K * group['u_D'].mean(),
            'decidable': decidable,
            'significant residuals': int(group['significant'].sum()),
            'mean pass difference': difference.mean(),
            'mean absolute pass difference (%)': np.nanmean(relative_difference),
            'max absolute pass difference (%)': np.nanmax(relative_difference),
        }
    return pd.DataFrame(rows).T


def relative_residuals(table: pd.DataFrame) -> pd.DataFrame:
    """Return the range of the relative residual in % and its correlation with the measured value."""
    relative = 100 * table['residual'] / table['measured']
    rows = {}
    for quantity, group in relative.groupby(table['quantity'], sort=False):
        measured = table.loc[group.index, 'measured']
        rows[quantity] = {
            'min (%)': group.min(),
            'max (%)': group.max(),
            'correlation with measured value': group.corr(measured),
        }
    return pd.DataFrame(rows).T


def evaluate_uncertainty(
    results: dict, conditions: pd.DataFrame, trigger: str, budget: dict, threshold: float = 3.0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the per-condition table and the summary of a stationary evaluation."""
    measured = results['measurement_conditions']
    simulated = results['simulation_conditions']
    u_b = absolute_type_b(measured, relative_type_b(budget, measured))
    u_a = type_a_of_conditions(results['u_A'], results['measurement'][trigger], conditions)
    table = comparison_table(measured, simulated, u_b, u_a)
    return table, summarize(table, results['pass_difference'], threshold)
