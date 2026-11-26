"""Tests of the time window of the CSV source."""

import pandas as pd
import pytest

from src.io.data_processing import _apply_time_window


def frame():
    return pd.DataFrame({'DateTime': ['2026-07-30 14:43:29', '2026-07-30 14:43:30', '2026-07-30 15:00:00',
                                      '2026-07-30 23:24:25', '2026-07-30 23:24:26'], 'x': [1, 2, 3, 4, 5]})


def test_both_bounds_are_included_like_the_database_query():
    result = _apply_time_window(frame(), '2026-07-30 14:43:30', '2026-07-30 23:24:25')
    assert result['x'].tolist() == [2, 3, 4]


def test_no_window_keeps_all_rows():
    assert len(_apply_time_window(frame(), None, None)) == 5


def test_empty_window_raises():
    with pytest.raises(ValueError):
        _apply_time_window(frame(), '2026-08-01', '2026-08-02')


def test_usecols_keeps_time_and_run_signals_only():
    from types import SimpleNamespace
    from src.io.data_processing import _csv_usecols
    config = SimpleNamespace(data_source=SimpleNamespace(conditions=None, input_variables=['a'],
                                                         output_variables=['b']))
    keep = _csv_usecols(config)
    assert [c for c in ['DateTime', 'a', 'b', 'c'] if keep(c)] == ['DateTime', 'a', 'b']


def test_usecols_reads_all_columns_with_conditions():
    from types import SimpleNamespace
    from src.io.data_processing import _csv_usecols
    config = SimpleNamespace(data_source=SimpleNamespace(conditions=['c > 0'], input_variables=['a'],
                                                         output_variables=['b']))
    assert _csv_usecols(config) is None


def test_measurement_csv_replaces_the_database(monkeypatch, tmp_path):
    from src.io.run_workflow import load_run_config
    monkeypatch.setenv('FCS_MEASUREMENT_CSV', str(tmp_path / 'raw.csv'))
    assert load_run_config('D0').data_source.csv_path == tmp_path / 'raw.csv'
    monkeypatch.delenv('FCS_MEASUREMENT_CSV')
    assert load_run_config('D0').data_source.csv_path is None
