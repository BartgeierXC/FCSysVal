import numpy as np
import pandas as pd
import pytest

import data.parameters.fcs_variable_naming as fcn
from src.io import coverage as cov


def make_steps(n_conditions: int = 3) -> pd.DataFrame:
    """Return valid steps in which each condition is run twice, in reverse order."""
    current = 150.0 + 20.0 * np.arange(n_conditions)
    current = np.r_[current, current[::-1]]
    df = pd.DataFrame({
        fcn.I_A_set: current,
        fcn.p_Si_A_set_bara: 1.65,
        fcn.p_Si_Ca_set_bara: 1.45,
        fcn.v_ARB_req_rpm: 5000.0,
        fcn.SS_Delta_T_degC: 4.0,
        fcn.SS_T_Stack_degC: 72.0,
        fcn.DPT_Si_Ca_degC_set: 49.0,
        fcn.profile_step_no: np.arange(1.0, len(current) + 1),
    })
    # Air flow of a stoichiometry of 2.25, derived with the inverse of to_stoic
    df[fcn.SS_vf_air_set_nl_per_min] = 2.25 / cov.pe.to_stoic(1.0, df[fcn.I_A_set], cov.N_CELLS)
    for i, column in enumerate(cov.MEASURED_COLUMNS):
        df[column] = 100.0 + i + np.arange(len(df))
    return df


def test_exact_rule_rejects_point_on_the_pressure_limit():
    point = pd.DataFrame({fcn.p_Si_A_set_bara: [1.42], fcn.p_Si_Ca_set_bara: [1.27 + 1e-12]})

    assert not cov.make_rules()[2](point).iloc[0]


def test_tolerance_accepts_rounded_setpoints():
    point = pd.DataFrame({fcn.SS_T_Stack_degC: [72.29], fcn.DPT_Si_Ca_degC_set: [46.57]})

    assert not cov.make_rules()[1](point).iloc[0]
    assert cov.make_rules(cov.RULE_TOLERANCE)[1](point).iloc[0]


def test_operating_conditions_merge_two_passes():
    design, measured, table = cov.operating_conditions(make_steps())

    assert len(design) == 3
    assert len(measured) == 3
    assert table.groupby('condition').size().eq(2).all()
    assert sorted(table['pass'].unique()) == [1, 2]


def test_operating_conditions_average_measured_passes():
    _, measured, table = cov.operating_conditions(make_steps())

    # Steps 1 and 6 of the first condition have the measured offsets 0 and 5.
    assert sorted(measured[cov.MEASURED_COLUMNS[0]]) == pytest.approx([102.5, 102.5, 102.5])
    assert table.loc[table['step'] == 6, 'pass'].iloc[0] == 2


def test_conditions_table_lists_d0_and_d1():
    steps = make_steps()
    design, measured, _ = cov.operating_conditions(steps)
    augmented = pd.concat([design, design.iloc[:2] + 1.0], ignore_index=True)
    study = cov.CoverageStudy(design, measured, None, None, None, None, None, None, [], [], augmented, 2)

    table = cov.conditions_table(steps, study)

    assert list(table['set']) == ['D0'] * 3 + ['D1'] * 2
    assert list(table['condition']) == [1, 2, 3, 4, 5]
    assert table.loc[table['set'] == 'D0', ['step pass 1', 'step pass 2']].notna().all().all()
    realized = [c for c in table.columns if ' realized ' in c]
    assert len(realized) == 8
    assert table.loc[table['set'] == 'D1', realized].isna().all().all()
    assert table.loc[table['set'] == 'D0', 'lambda_C_set planned [-]'].to_numpy() == pytest.approx(2.25)
    # The first condition holds steps 1 and 6, its realized current is the mean of both passes.
    first = table.iloc[0]
    assert (first['step pass 1'], first['step pass 2']) == (1, 6)
    assert first['I_set realized [A]'] == pytest.approx(102.5)


def test_operating_conditions_reject_single_pass():
    steps = make_steps().iloc[:-1]

    with pytest.raises(ValueError):
        cov.operating_conditions(steps)
