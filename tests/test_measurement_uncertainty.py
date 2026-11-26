import numpy as np
import pandas as pd
import pytest

import data.parameters.fcs_variable_naming as fcn
from src.io import measurement_uncertainty as mu

BUDGET = {
    'chain': {'reading': 0.1, 'conditioner': 0.1, 'adc': 0.031},
    'signals': {
        fcn.U_cell_V: {'limits': [0.6], 'chain': []},
        fcn.P_CMP_kW: {'limits': [0.1], 'chain': ['conditioner', 'adc']},
        fcn.P_ARB_kW: {'limits': [1.0], 'chain': ['conditioner', 'adc']},
        fcn.P_CWP_W: {'limits': [2.5], 'chain': ['conditioner', 'adc']},
        fcn.I_A_act: {'limits': [0.5]},
    },
}


def make_values() -> pd.DataFrame:
    df = pd.DataFrame({
        fcn.U_cell_V: [0.7],
        fcn.P_el_kW: [100.0],
        fcn.P_CMP_kW: [8.0],
        fcn.P_ARB_kW: [1.0],
        fcn.P_CWP_W: [1000.0],
        fcn.I_A_act: [200.0],
    })
    df[fcn.P_sys_kW] = 90.0
    df[fcn.eta_sys_pct] = 50.0
    return df


def test_relative_type_b_reproduces_the_sensor_budget():
    result = mu.relative_type_b(BUDGET)

    assert result[fcn.U_cell_V] == pytest.approx(0.346, abs=1e-3)
    assert result[fcn.P_CMP_kW] == pytest.approx(0.084, abs=1e-3)
    assert result[fcn.P_CWP_W] == pytest.approx(1.445, abs=1e-3)


def test_stack_power_combines_cell_voltage_and_current():
    relative = mu.relative_type_b(BUDGET)

    result = mu.absolute_type_b(make_values(), relative)

    expected = 100.0 * np.hypot(relative[fcn.U_cell_V], relative[fcn.I_A_act]) / 100
    assert result[fcn.P_el_kW].iloc[0] == pytest.approx(expected)


def test_system_power_adds_the_absolute_contributions():
    values = make_values()
    relative = mu.relative_type_b(BUDGET)

    result = mu.absolute_type_b(values, relative)

    parts = [100.0 * np.hypot(relative[fcn.U_cell_V], relative[fcn.I_A_act]), 8.0 * relative[fcn.P_CMP_kW],
             1.0 * relative[fcn.P_ARB_kW], 1.0 * relative[fcn.P_CWP_W]]
    assert result[fcn.P_sys_kW].iloc[0] == pytest.approx(np.sqrt(sum(p ** 2 for p in parts)) / 100)


def test_efficiency_propagates_cell_voltage_auxiliaries_and_current_once():
    values = make_values()
    relative = mu.relative_type_b(BUDGET)

    result = mu.absolute_type_b(values, relative)

    # eta = U_c / U_LHV - P_aux / (N I U_LHV), the current enters only through the auxiliary term.
    lhv_power = mu.N_CELLS * 200.0 * 1.253 / 1e3
    u_cell = 0.7 * relative[fcn.U_cell_V] / 100
    u_aux = [8.0 * relative[fcn.P_CMP_kW] / 100, 1.0 * relative[fcn.P_ARB_kW] / 100,
             1.0 * relative[fcn.P_CWP_W] / 100]
    u_current = 200.0 * relative[fcn.I_A_act] / 100
    expected = 100 * np.sqrt((u_cell / 1.253) ** 2 + sum((x / lhv_power) ** 2 for x in u_aux)
                             + (10.0 * u_current / (lhv_power * 200.0)) ** 2)
    assert result[fcn.eta_sys_pct].iloc[0] == pytest.approx(expected)


def test_efficiency_uncertainty_is_dominated_by_the_cell_voltage():
    result = mu.absolute_type_b(make_values(), mu.relative_type_b(BUDGET))

    u_cell_only = 100 * 0.7 * mu.relative_type_b(BUDGET)[fcn.U_cell_V] / 100 / 1.253
    assert result[fcn.eta_sys_pct].iloc[0] == pytest.approx(u_cell_only, rel=0.05)


def test_type_a_of_two_passes_is_half_of_the_quadratic_sum():
    u_a = pd.DataFrame({fcn.P_sys_kW: [3.0, 4.0]}, index=[10.0, 20.0])
    steps = pd.Series([1.0, 2.0], index=u_a.index)
    conditions = pd.DataFrame({'step': [1, 2], 'condition': [1, 1], 'pass': [1, 2]})

    result = mu.type_a_of_conditions(u_a, steps, conditions)

    assert result[fcn.P_sys_kW].loc[1] == pytest.approx(2.5)


def test_residual_is_significant_above_twice_the_uncertainty():
    values = pd.concat([make_values()] * 2, ignore_index=True)
    simulated = values.copy()
    simulated.loc[0, fcn.P_sys_kW] = 90.0 - 2.1
    simulated.loc[1, fcn.P_sys_kW] = 90.0 - 1.9
    u_b = pd.DataFrame(0.0, index=values.index, columns=values.columns)
    u_b[fcn.P_sys_kW] = 1.0
    u_a = u_b * 0

    table = mu.comparison_table(values, simulated, u_b, u_a)

    rows = table[table['quantity'] == fcn.P_sys_kW]
    assert rows['significant'].tolist() == [True, False]
    assert rows['residual'].iloc[0] == pytest.approx(2.1)


def test_full_scale_limit_is_converted_to_the_reading():
    budget = {
        'fixed': {'cmp_voltage': 400, 'arb_voltage': 24},
        'signals': {
            fcn.P_ARB_kW: {'full_scale': [{'percent': 0.2, 'of': 170, 'reading': 'arb_current'}]},
            fcn.U_cell_V: {'limits': [0.6]},
        },
    }
    values = pd.DataFrame({fcn.P_ARB_kW: [2.04, 0.204], fcn.U_cell_V: [0.7, 0.7],
                           fcn.P_el_kW: [1.0, 1.0], fcn.P_CMP_kW: [1.0, 1.0]})

    result = mu.relative_type_b(budget, values)

    # 2.04 kW at 24 V is 85 A, half of the nominal 170 A, so the limit doubles in relative terms.
    assert result[fcn.P_ARB_kW].iloc[0] == pytest.approx(0.4 / np.sqrt(3))
    assert result[fcn.P_ARB_kW].iloc[1] == pytest.approx(4.0 / np.sqrt(3))
    assert result[fcn.U_cell_V].iloc[0] == pytest.approx(0.6 / np.sqrt(3))


def test_full_scale_limits_need_values():
    budget = {'fixed': {'cmp_voltage': 400, 'arb_voltage': 24},
              'signals': {fcn.P_ARB_kW: {'full_scale': [{'percent': 0.2, 'of': 170, 'reading': 'arb_current'}]}}}
    with pytest.raises(ValueError):
        mu.relative_type_b(budget)


def test_absolute_and_linear_limits_relative_to_reading():
    budget = {'fixed': {}, 'signals': {
        'p': {'absolute': [0.02]},
        'rh': {'linear': [{'absolute': 1.5, 'percent': 1.5}]},
    }}
    values = pd.DataFrame({'p': [2.0, 1.0], 'rh': [30.0, 50.0]})
    result = mu.relative_type_b(budget, values)
    np.testing.assert_allclose(result['p'], [1.0 / np.sqrt(3), 2.0 / np.sqrt(3)])
    np.testing.assert_allclose(result['rh'], [(5.0 + 1.5) / np.sqrt(3), (3.0 + 1.5) / np.sqrt(3)])


def test_absolute_limit_needs_values():
    with pytest.raises(ValueError):
        mu.relative_type_b({"fixed": {}, "signals": {"p": {"absolute": [0.02]}}})
