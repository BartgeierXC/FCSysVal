"""Workflow steps of a validation run, called one by one in the run notebooks."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

import data.parameters.fcs_variable_naming as fcn
import src.utils.physical_equations as pe
from src.io import config_interface as ci
from src.io import data_processing as dp
from src.io.runtime_paths import (
    map_nested_sim_config,
    resolve_config_path,
    resolve_download_dir,
    resolve_lut_path,
    resolve_measurement_csv,
    resolve_simulation_root,
)
from src.utils.constants import U_LHV
from src.utils.om_util import import_mat_result

N_CELLS = 275  # number of cells of the stack
# Power of the electronic load, kept as a diagnostic quantity. The evaluated stack power is N I U_c.
P_LOAD_kW = 'P.S.act load [kW]'

# Factors from the SI units of the model to the units of the measured signals.
SIM_FACTORS = {
    'U.S.AveCell [V]': 1,
    'P.S.act [W]': 1e-3,
    'P.CMP.CV [kW]': 1e-3,
    'P.ARB.CV [kW]': 1e-3,
    'P.FU [W]': 1,
    'RH.Si.C [%]': 100,
    'T.Hu.i [°C]': 1,
    'T.Si.C [°C]': 1,
    'T.So.C [°C]': 1,
    'T.Si.CL [°C]': 1,
    'T.So.CL [°C]': 1,
    'T.CMP.o [°C]': 1,
    'p.So.A [bara]': 1e-5,
    'p.So.C [bara]': 1e-5,
    'pD.S.C [mbar]': 1e-2,
    'pD.S.CL.CV [mbar]': 1e-2,
    'F.Si.Cmp.C [kg/h]': 3.6e3,
    'F.HF.HE.CL [l/min]': 6e4,
    'p.CMP.ratio.CV [-]': 1,
    'p.Si.C [bara]': 1e-5,
    'Profile_StepNo [-]': 1,
}
TEMPERATURE_COLUMNS = (
    fcn.T_Si_Ca_degC, fcn.T_So_Ca_degC, fcn.T_Si_CL_degC,
    fcn.T_So_CL_degC, fcn.T_Hu_i_degC, fcn.T_CMP_o_degC,
)


def load_run_config(run_id: str) -> ci.ValidationConfig:
    """Return the validation config of *run_id* with the local paths applied.

    The measurements are read from the file given by FCS_MEASUREMENT_CSV.
    """
    config = ci.ValidationConfig.from_yaml(str(resolve_config_path(f'run_{run_id}.yml')))
    config.sim_config.config_path = map_nested_sim_config(config.sim_config.config_path)
    config.sim_config.input_path = resolve_lut_path()
    measurement_csv = resolve_measurement_csv()
    if measurement_csv is not None:
        config.data_source.csv_path = measurement_csv
    return config


def load_measurements(config: ci.ValidationConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the measured inputs and outputs of a run on its time axis."""
    return dp.get_data_source(config)


def prepare_run_input(run_id: str) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Load the measured data of *run_id* and write the simulation input table.

    Returns the measured inputs, the measured outputs and the stop time in s.
    """
    config = load_run_config(run_id)
    input_df, output_df = load_measurements(config)
    dp.prepare_sim_input(config, simulation_table(input_df))
    return input_df, output_df, float(input_df.index.max())


def simulation_table(input_df: pd.DataFrame) -> pd.DataFrame:
    """Return the measured inputs as they enter the simulation input table."""
    table_df = input_df.copy()
    if fcn.vf_air_Ca_nl_per_min in table_df:
        # The MFC air subsystem takes the air mass flow instead of the compressor speed.
        table_df[fcn.vf_air_Ca_nl_per_min] = pe.from_nl_to_mf_air(table_df[fcn.vf_air_Ca_nl_per_min])
    return table_df


PARAMETER_SETS = ('iteration1', 'iteration2')


def published_result(run_id: str, parameter_set: str) -> Path:
    """Return the result file of *run_id* in the record layout <root>/<parameter_set>/<run_id>/output/."""
    if parameter_set not in PARAMETER_SETS:
        raise ValueError(f'Unknown parameter set {parameter_set}, expected one of {PARAMETER_SETS}')
    matches = sorted((resolve_simulation_root() / parameter_set / run_id / 'output').glob('*_res*.mat'))
    if len(matches) != 1:
        raise FileNotFoundError(f'Expected one result of run {run_id} for {parameter_set}, found {len(matches)}')
    return matches[0]


def latest_result(run_id: str) -> Path:
    """Return the result file of the latest stored attempt of *run_id*."""
    matches = sorted(resolve_simulation_root().glob(f'*/{run_id}_*/r[0-9][0-9]/output/*_res*.mat'))
    if not matches:
        raise FileNotFoundError(f'No stored result of run {run_id}')
    return matches[-1]


def stored_run_summary(result_path: Path) -> pd.Series:
    """Return the attempt, solver settings, model commit and note of a stored result."""
    manifest = json.loads((result_path.parents[1] / 'manifest.json').read_text(encoding='utf-8'))
    simulation = manifest['simulation']
    return pd.Series({
        'Attempt': manifest['attempt'],
        'Stored': manifest['stored'],
        'Solver': simulation['solver'],
        'Tolerance': simulation['tolerance'],
        'Output interval [s]': simulation['stepSize'],
        'Stop time [s]': simulation['stopTime'],
        'VFCS commit': manifest['vfcs_commit_argument'],
        'Note': manifest['note'],
    })


def load_simulation(config: ci.ValidationConfig, result_path: Path, step_size: float) -> pd.DataFrame:
    """Return the simulated outputs in the units of the measured signals."""
    sim_config = dp.parse_config(str(config.sim_config.config_path))
    columns = sim_config['simulation_setup']['output_configuration']['parameter_names']
    data = import_mat_result(str(result_path), columns, timestep=step_size)
    return convert_simulation(dp.first_sim_to_df(config, [[0, data.T]]))


def convert_simulation(sim_df: pd.DataFrame) -> pd.DataFrame:
    """Convert simulated signals to the units of the measured signals."""
    sim_df = sim_df.mul(pd.Series(SIM_FACTORS)[sim_df.columns], axis=1)
    sim_df[fcn.rh_Si_Ca_pct] = sim_df[fcn.rh_Si_Ca_pct].clip(upper=100)
    for column in TEMPERATURE_COLUMNS:
        if column in sim_df:
            sim_df[column] = pe.to_Celsius(sim_df[column])
    return sim_df.rename(columns={'P.S.act [W]': fcn.P_el_kW})


def convert_source(source_df: pd.DataFrame) -> pd.DataFrame:
    """Express the power of the electronic load as a positive power in kW."""
    source_df = source_df.rename(columns={'P.S.act [W]': P_LOAD_kW})
    source_df[P_LOAD_kW] = -1e-3 * source_df[P_LOAD_kW]
    return source_df


def add_system_quantities(df: pd.DataFrame) -> pd.DataFrame:
    """Add the system power and the system efficiency, the compressor is optional.

    The stack power is P_FC = N I U_c and the system efficiency eta_sys = P_sys / (N I U_LHV), for
    measurement and simulation alike. The measured frame carries the stack current I of the electronic
    load and the mean cell voltage U_c of the cell voltage monitoring. The model has no current output,
    its current follows from its stack power P_FC = N I U_c. The measured power of the electronic load
    is kept as a diagnostic quantity, the simulation gives it as its stack power.
    """
    df = df.copy()
    if fcn.I_A_act in df:
        # The stack power takes the column position of the load power, which moves to the end.
        df.insert(list(df.columns).index(P_LOAD_kW), fcn.P_el_kW,
                  N_CELLS * df[fcn.I_A_act] * df[fcn.U_cell_V] / 1e3)
        load = df.pop(P_LOAD_kW)
    else:
        load = df[fcn.P_el_kW]
    p_aux = df[fcn.P_ARB_kW] + df[fcn.P_CWP_W] / 1000
    if fcn.P_CMP_kW in df:
        p_aux = p_aux + df[fcn.P_CMP_kW]
    df[fcn.P_sys_kW] = df[fcn.P_el_kW] - p_aux
    if fcn.I_A_act in df:
        current = df[fcn.I_A_act]
    else:
        current = df[fcn.P_el_kW] * 1e3 / (N_CELLS * df[fcn.U_cell_V])
    eta = 100 * df[fcn.P_sys_kW] * 1e3 / (N_CELLS * current * U_LHV)
    df[fcn.eta_sys_pct] = eta.clip(0, 100)
    df[P_LOAD_kW] = load
    return df


def sample_on_source_time(
    sim_df: pd.DataFrame, source_df: pd.DataFrame, trigger: str | None = None
) -> pd.DataFrame:
    """Return the simulation at the measured times, using the nearest output sample.

    The measured samples are not equidistant, so both signals are compared on
    the measured time base. With *trigger*, the measured trigger column replaces
    the simulated one, so that both are averaged over identical windows.
    """
    sampled_df = sim_df.sort_index().reindex(source_df.index, method='nearest')
    if trigger is not None:
        sampled_df[trigger] = source_df[trigger]
    return sampled_df


def load_comparison(
    run_id: str, step_size: float, result_path: Path | None = None
) -> tuple[ci.ValidationConfig, pd.DataFrame, pd.DataFrame]:
    """Return simulation and measurement of a run on the measured time base, after warm-up."""
    config = load_run_config(run_id)
    input_df, source_df = load_measurements(config)
    # The stack current is a simulation input. The measured system efficiency refers to it.
    source_df = source_df.assign(**{fcn.I_A_act: input_df[fcn.I_A_act].reindex(source_df.index)})
    sim_df = load_simulation(config, result_path or latest_result(run_id), step_size)
    sim_df, source_df = align_comparison(config, sim_df, source_df)
    skip_time = config.data_source.skip_time
    return config, sim_df.loc[skip_time:], source_df.loc[skip_time:]


def align_comparison(
    config: ci.ValidationConfig, sim_df: pd.DataFrame, source_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return simulation and measurement with system quantities on the measured time base."""
    source_df = add_system_quantities(convert_source(source_df))
    sim_df = add_system_quantities(sim_df)
    trigger = config.group_config.trigger if config.group_config else None
    return sample_on_source_time(sim_df, source_df, trigger), source_df


def average_passes(groups: pd.DataFrame, conditions: pd.DataFrame, trigger: str) -> pd.DataFrame:
    """Return the mean over the passes of each operating condition, indexed by condition."""
    step_to_condition = conditions.set_index('step')['condition']
    selected = groups[groups[trigger].round().astype(int).isin(step_to_condition.index)]
    labels = selected[trigger].round().astype(int).map(step_to_condition).to_numpy()
    return selected.groupby(labels).mean().rename_axis('condition')


def pass_difference(groups: pd.DataFrame, conditions: pd.DataFrame, trigger: str) -> pd.DataFrame:
    """Return second minus first pass of each operating condition, indexed by condition."""
    steps = groups.set_index(groups[trigger].round().astype(int)).drop(columns=trigger)
    first = conditions[conditions['pass'] == 1].set_index('condition')['step']
    second = conditions[conditions['pass'] == 2].set_index('condition')['step']
    return steps.loc[second].set_axis(second.index) - steps.loc[first].set_axis(first.index)


def pass_groups(groups: pd.DataFrame, conditions: pd.DataFrame, trigger: str, number: int) -> pd.DataFrame:
    """Return the windows of one pass, indexed by condition."""
    steps = groups.set_index(groups[trigger].round().astype(int)).drop(columns=trigger)
    selected = conditions[conditions['pass'] == number].set_index('condition')['step']
    return steps.loc[selected].set_axis(selected.index)


def pass_comparison(results: dict, conditions: pd.DataFrame, trigger: str) -> pd.DataFrame:
    """Return the mean second minus first pass of experiment and simulation per signal.

    The correlation is taken between the measured and simulated differences over the conditions.
    """
    # The measured stack current is only an input of the system efficiency, it has no simulated counterpart.
    measured = results['pass_difference'].drop(columns=fcn.I_A_act, errors='ignore')
    simulated = pass_difference(results['simulation'], conditions, trigger)
    return pd.DataFrame({
        'mean measured difference': measured.mean(),
        'mean simulated difference': simulated.mean(),
        'correlation': measured.corrwith(simulated),
    })


def mre_per_pass(results: dict, conditions: pd.DataFrame, trigger: str) -> pd.DataFrame:
    """Return the mean relative error in % of each signal, separately for each pass."""
    return pd.DataFrame({
        f'pass {number}': dp.calc_err_metrics(
            pass_groups(results['simulation'], conditions, trigger, number),
            pass_groups(results['measurement'], conditions, trigger, number),
        )['MRE (%)']
        for number in (1, 2)
    })


def numerical_difference(reference: dict, finer: dict, trigger: str) -> pd.DataFrame:
    """Return the difference of two evaluations of one run that differ in the solver tolerance.

    The relative difference refers to *reference*, per operating condition.
    """
    simulated = reference['simulation_conditions'].drop(columns=trigger)
    difference = (finer['simulation_conditions'].drop(columns=trigger) - simulated).abs()
    return pd.DataFrame({
        'max absolute difference': difference.max(),
        'max relative difference (%)': (100 * difference / simulated.abs()).max(),
        'MRE reference (%)': reference['metrics']['MRE (%)'].reindex(simulated.columns),
        'MRE finer (%)': finer['metrics']['MRE (%)'].reindex(simulated.columns),
    })


def stationary_details(results: dict, conditions: pd.DataFrame, trigger: str) -> dict[str, pd.DataFrame]:
    """Return the tables behind the Result Evaluation text, keyed by name."""
    return {
        'window_std': results['std'].drop(columns=[trigger, fcn.I_A_act], errors='ignore').agg(['max', 'median']).T.rename(
            columns={'max': 'sigma_max', 'median': 'sigma_median'}
        ),
        'pass_comparison': pass_comparison(results, conditions, trigger),
        'mre_per_pass': mre_per_pass(results, conditions, trigger),
    }


def evaluate_stationary(run_id: str, step_size: float = 1, result_path: Path | None = None) -> dict:
    """Return the group means, sample standard deviations and error metrics of a run.

    With a conditions file, the metrics compare the means over the passes of each condition.
    """
    config, sim_df, source_df = load_comparison(run_id, step_size, result_path)
    return compare_stationary(config, sim_df, source_df)


def compare_stationary(config: ci.ValidationConfig, sim_df: pd.DataFrame, source_df: pd.DataFrame) -> dict:
    """Return the group means, sample standard deviations and error metrics after warm-up.

    With a conditions file, the metrics compare the means over the passes of each condition.
    """
    sim_groups = dp.group_trigger_data(config, sim_df, reset_index=False)
    source_groups, source_std = dp.group_trigger_data(
        config, source_df, reset_index=False, return_std=True
    )
    group_time = config.group_config.group_time
    n_samples = pd.Series(
        [((source_df.index > t - group_time) & (source_df.index <= t)).sum() for t in source_groups.index],
        index=source_groups.index,
    )
    results = {
        'simulation': sim_groups,
        'measurement': source_groups,
        'std': source_std,
        'u_A': source_std.div(np.sqrt(n_samples), axis=0),
        'n_samples': n_samples,
    }
    if config.group_config.conditions_file:
        trigger = config.group_config.trigger
        conditions = pd.read_csv(resolve_config_path(config.group_config.conditions_file))
        sim_groups = average_passes(sim_groups, conditions, trigger)
        source_groups = average_passes(source_groups, conditions, trigger)
        results['pass_difference'] = pass_difference(results['measurement'], conditions, trigger)
        results['simulation_conditions'] = sim_groups
        results['measurement_conditions'] = source_groups
    results['metrics'] = dp.calc_err_metrics(sim_groups, source_groups)
    return results


def evaluate_transient(run_id: str, step_size: float = 0.01, result_path: Path | None = None) -> dict:
    """Return the transient error metrics and error time series of a run."""
    _, sim_df, source_df = load_comparison(run_id, step_size, result_path)
    return compare_transient(sim_df, source_df)


def compare_transient(sim_df: pd.DataFrame, source_df: pd.DataFrame) -> dict:
    """Return the transient error metrics and error time series after warm-up."""
    columns = [c for c in sim_df.columns if c in source_df.columns and c != fcn.profile_step_no]
    metrics, errors = dp.calc_transient_err_metrics(
        sim_df[columns], source_df[columns], return_timeseries=True
    )
    return {'simulation': sim_df, 'measurement': source_df, 'metrics': metrics, 'errors': errors}
