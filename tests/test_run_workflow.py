from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.io import data_processing as dp
from src.io.run_workflow import sample_on_source_time


def make_config(group_time: int, trigger_steps: int) -> SimpleNamespace:
    group_config = SimpleNamespace(
        group_time=group_time, trigger_steps=trigger_steps, trigger='step', tolerance=0.1
    )
    return SimpleNamespace(group_config=group_config)


def make_steps(step_values: list[float], rows: int, dt: float) -> pd.DataFrame:
    """Return consecutive steps with *rows* samples each, spaced by *dt* seconds."""
    step = np.repeat(np.arange(len(step_values), dtype=float), rows)
    value = np.repeat(step_values, rows)
    return pd.DataFrame({'step': step, 'y': value}, index=np.arange(len(step)) * dt)


def test_elapsed_time_index_uses_seconds_since_first_sample():
    df = pd.DataFrame({
        'DateTime': ['2026-07-30 14:43:30.207', '2026-07-30 14:43:31.307', '2026-07-30 14:43:31.507']
    })

    result = dp._apply_elapsed_time_index(df, 'elapsed')

    assert result.index.tolist() == pytest.approx([0.0, 1.1, 1.3])


def test_elapsed_time_index_keeps_rows_without_time_axis():
    df = pd.DataFrame({'DateTime': ['2026-07-30 14:43:30']})

    assert dp._apply_elapsed_time_index(df, None) is df


def test_elapsed_time_index_rejects_unknown_axis():
    df = pd.DataFrame({'DateTime': ['2026-07-30 14:43:30']})

    with pytest.raises(ValueError):
        dp._apply_elapsed_time_index(df, 'rows')


def test_group_averages_last_group_time_seconds_of_step():
    df = make_steps([1.0, 2.0, 3.0], rows=280, dt=1.1)
    # Only the last 150 s of the first step carry the settled value.
    settled = df.index[df['step'] == 0][-1] - 150
    df.loc[(df['step'] == 0) & (df.index <= settled), 'y'] = 100.0

    grouped = dp.group_trigger_data(make_config(150, 275), df)

    assert grouped['y'].iloc[0] == pytest.approx(1.0)


def test_group_window_is_not_limited_by_trigger_steps():
    df = make_steps([1.0, 2.0, 3.0], rows=280, dt=1.0)
    df['y'] = np.arange(len(df), dtype=float)

    grouped = dp.group_trigger_data(make_config(150, 275), df)

    # Mean of rows 130 to 279, not of the 6 rows after trigger_steps.
    assert grouped['y'].iloc[0] == pytest.approx(np.arange(130, 280).mean())


def test_group_skips_steps_shorter_than_trigger_steps():
    long_steps = make_steps([1.0, 2.0], rows=280, dt=1.0)
    short_step = make_steps([9.0], rows=100, dt=1.0)
    short_step['step'] = 5.0
    short_step.index = short_step.index + 280
    last = long_steps.iloc[280:].copy()
    last.index = last.index + 100
    df = pd.concat([long_steps.iloc[:280], short_step, last])

    grouped = dp.group_trigger_data(make_config(150, 275), df)

    assert grouped['y'].tolist() == pytest.approx([1.0, 2.0])


def test_sample_on_source_time_uses_measured_time_base():
    sim_df = pd.DataFrame({'y': [0.0, 1.0, 2.0, 3.0]}, index=[0.0, 1.0, 2.0, 3.0])
    source_df = pd.DataFrame({'y': [5.0, 6.0]}, index=[0.9, 2.2])

    result = sample_on_source_time(sim_df, source_df)

    assert result.index.tolist() == [0.9, 2.2]
    assert result['y'].tolist() == [1.0, 2.0]


def test_sample_on_source_time_takes_measured_trigger():
    sim_df = pd.DataFrame({'y': [0.0, 1.0], 'step': [7.0, 7.0]}, index=[0.0, 1.0])
    source_df = pd.DataFrame({'y': [0.0, 1.0], 'step': [1.0, 2.0]}, index=[0.0, 1.0])

    result = sample_on_source_time(sim_df, source_df, trigger='step')

    assert result['step'].tolist() == [1.0, 2.0]
