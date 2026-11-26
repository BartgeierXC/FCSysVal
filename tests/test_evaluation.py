import numpy as np
import pandas as pd
import pytest

import data.parameters.fcs_variable_naming as fcn
import src.utils.physical_equations as pe
from src.io import run_workflow as rw


def make_outputs(with_cmp: bool) -> pd.DataFrame:
    df = pd.DataFrame({
        fcn.U_cell_V: [0.7],
        fcn.P_el_kW: [100.0],
        fcn.P_ARB_kW: [2.0],
        fcn.P_CWP_W: [1000.0],
    })
    if with_cmp:
        df[fcn.P_CMP_kW] = [7.0]
    return df


def test_system_power_subtracts_all_auxiliaries():
    result = rw.add_system_quantities(make_outputs(with_cmp=True))

    assert result[fcn.P_sys_kW].iloc[0] == pytest.approx(90.0)


def test_system_power_without_compressor():
    result = rw.add_system_quantities(make_outputs(with_cmp=False))

    assert result[fcn.P_sys_kW].iloc[0] == pytest.approx(97.0)


def test_system_efficiency_is_system_power_over_lhv_power():
    result = rw.add_system_quantities(make_outputs(with_cmp=True))

    # Without a current column, the model relation N I = P_FC / U_c gives the current.
    expected = pe.calc_eta(0.7) * 90.0 / 100.0
    assert result[fcn.eta_sys_pct].iloc[0] == pytest.approx(expected)


def make_measured() -> pd.DataFrame:
    """Measured frame: load power instead of the stack power, with the stack current."""
    return pd.DataFrame({
        fcn.U_cell_V: [0.7],
        rw.P_LOAD_kW: [76.0],
        fcn.P_CMP_kW: [7.0],
        fcn.P_ARB_kW: [2.0],
        fcn.P_CWP_W: [1000.0],
        fcn.I_A_act: [400.0],
    })


def test_measured_stack_power_is_cells_times_current_times_cell_voltage():
    result = rw.add_system_quantities(make_measured())

    stack_power = rw.N_CELLS * 400.0 * 0.7 / 1e3
    assert result[fcn.P_el_kW].iloc[0] == pytest.approx(stack_power)
    assert result[fcn.P_sys_kW].iloc[0] == pytest.approx(stack_power - 10.0)
    assert result[rw.P_LOAD_kW].iloc[0] == pytest.approx(76.0)


def test_measured_stack_power_keeps_the_column_position_of_the_load_power():
    result = rw.add_system_quantities(make_measured())

    assert list(result.columns)[1] == fcn.P_el_kW
    assert list(result.columns)[-1] == rw.P_LOAD_kW


def test_system_efficiency_uses_the_measured_current():
    result = rw.add_system_quantities(make_measured())

    p_sys = rw.N_CELLS * 400.0 * 0.7 / 1e3 - 10.0
    expected = 100 * p_sys * 1e3 / (rw.N_CELLS * 400.0 * 1.253)
    assert result[fcn.eta_sys_pct].iloc[0] == pytest.approx(expected)


def test_simulated_load_power_equals_the_stack_power():
    result = rw.add_system_quantities(make_outputs(with_cmp=True))

    assert result[rw.P_LOAD_kW].iloc[0] == pytest.approx(100.0)


def test_convert_source_gives_positive_power_in_kw():
    source_df = pd.DataFrame({'P.S.act [W]': [-100000.0]})

    assert rw.convert_source(source_df)[rw.P_LOAD_kW].iloc[0] == pytest.approx(100.0)


def test_convert_simulation_uses_measured_units():
    sim_df = pd.DataFrame({
        'P.S.act [W]': [100000.0],
        'RH.Si.C [%]': [1.2],
        'T.Si.CL [°C]': [343.15],
        'p.So.C [bara]': [150000.0],
        'F.Si.Cmp.C [kg/h]': [0.02],
    })

    result = rw.convert_simulation(sim_df)

    assert result[fcn.P_el_kW].iloc[0] == pytest.approx(100.0)
    assert result['RH.Si.C [%]'].iloc[0] == pytest.approx(100.0)
    assert result['T.Si.CL [°C]'].iloc[0] == pytest.approx(70.0)
    assert result['p.So.C [bara]'].iloc[0] == pytest.approx(1.5)
    assert result['F.Si.Cmp.C [kg/h]'].iloc[0] == pytest.approx(72.0)


def test_convert_simulation_rejects_unknown_signal():
    with pytest.raises(KeyError):
        rw.convert_simulation(pd.DataFrame({'unknown': [1.0]}))


def test_air_flow_conversion_matches_normal_density():
    assert pe.from_nl_to_mf_air(1000) == pytest.approx(1000 / 60000 * 1.2925, rel=2e-3)


def test_latest_result_takes_highest_attempt(tmp_path, monkeypatch):
    for attempt in ('r01', 'r02'):
        output = tmp_path / 'stationaer' / 'D0_run' / attempt / 'output'
        output.mkdir(parents=True)
        (output / 'TestFuelCellSystem_res.mat').touch()
    monkeypatch.setattr(rw, 'resolve_simulation_root', lambda: tmp_path)

    assert rw.latest_result('D0').parts[-3] == 'r02'


def test_latest_result_raises_without_result(tmp_path, monkeypatch):
    monkeypatch.setattr(rw, 'resolve_simulation_root', lambda: tmp_path)

    with pytest.raises(FileNotFoundError):
        rw.latest_result('D0')


def make_passes() -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = pd.DataFrame({'step': [1.0, 2.0, 3.0, 4.0, 5.0], 'y': [10.0, 20.0, 99.0, 22.0, 14.0]})
    conditions = pd.DataFrame({
        'step': [1, 2, 4, 5], 'condition': [1, 2, 2, 1], 'pass': [1, 1, 2, 2],
    })
    return groups, conditions


def test_average_passes_keeps_only_listed_steps():
    groups, conditions = make_passes()

    result = rw.average_passes(groups, conditions, 'step')

    assert result['y'].tolist() == pytest.approx([12.0, 21.0])


def test_pass_difference_is_second_minus_first():
    groups, conditions = make_passes()

    result = rw.pass_difference(groups, conditions, 'step')

    assert result.loc[1, 'y'] == pytest.approx(4.0)
    assert result.loc[2, 'y'] == pytest.approx(2.0)
