"""Tests of the tables behind the Result Evaluation text."""

import numpy as np
import pandas as pd
import pytest

from src.io.measurement_uncertainty import relative_residuals
from src.io.run_workflow import (
    mre_per_pass,
    numerical_difference,
    pass_comparison,
    pass_groups,
    stationary_details,
)

TRIGGER = 'step'
CONDITIONS = pd.DataFrame({
    'step': [1, 2, 3, 4, 5, 6],
    'condition': [1, 2, 3, 3, 2, 1],
    'pass': [1, 1, 1, 2, 2, 2],
})


def groups(values):
    return pd.DataFrame({TRIGGER: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 'x': values})


def results():
    measurement = groups([10.0, 20.0, 30.0, 31.0, 22.0, 13.0])
    simulation = groups([11.0, 22.0, 33.0, 33.0, 22.0, 11.0])
    second = pass_groups(measurement, CONDITIONS, TRIGGER, 2)
    first = pass_groups(measurement, CONDITIONS, TRIGGER, 1)
    return {
        'measurement': measurement,
        'simulation': simulation,
        'pass_difference': second - first,
        'std': groups([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
    }


def test_pass_groups_returns_the_windows_of_one_pass_by_condition():
    second = pass_groups(results()['measurement'], CONDITIONS, TRIGGER, 2)
    assert second['x'].to_dict() == {3: 31.0, 2: 22.0, 1: 13.0}


def test_pass_comparison_means_second_minus_first_pass():
    table = pass_comparison(results(), CONDITIONS, TRIGGER)
    assert table.loc['x', 'mean measured difference'] == pytest.approx(2.0)
    assert table.loc['x', 'mean simulated difference'] == pytest.approx(0.0)


def test_mre_per_pass_uses_only_the_windows_of_each_pass():
    table = mre_per_pass(results(), CONDITIONS, TRIGGER)
    first = np.mean([1 / 10, 2 / 20, 3 / 30]) * 100
    assert table.loc['x', 'pass 1'] == pytest.approx(first)
    assert table.loc['x', 'pass 2'] != pytest.approx(first)


def test_window_std_takes_the_largest_window():
    details = stationary_details(results(), CONDITIONS, TRIGGER)
    assert details['window_std'].loc['x', 'sigma_max'] == pytest.approx(0.6)


def test_relative_residuals_are_grouped_by_quantity():
    table = pd.DataFrame({
        'quantity': ['a', 'a', 'a', 'b'],
        'measured': [10.0, 20.0, 40.0, 5.0],
        'residual': [-1.0, 0.0, 4.0, 1.0],
    })
    result = relative_residuals(table)
    assert result.loc['a', 'min (%)'] == pytest.approx(-10.0)
    assert result.loc['a', 'max (%)'] == pytest.approx(10.0)
    assert result.loc['a', 'correlation with measured value'] > 0.9


def test_numerical_difference_is_relative_to_the_reference():
    metrics = pd.DataFrame({'MRE (%)': {'x': 1.0}})
    reference = {'simulation_conditions': pd.DataFrame({TRIGGER: [1, 2], 'x': [100.0, 200.0]}), 'metrics': metrics}
    finer = {'simulation_conditions': pd.DataFrame({TRIGGER: [1, 2], 'x': [101.0, 199.0]}), 'metrics': metrics}
    table = numerical_difference(reference, finer, TRIGGER)
    assert table.loc['x', 'max absolute difference'] == pytest.approx(1.0)
    assert table.loc['x', 'max relative difference (%)'] == pytest.approx(1.0)
