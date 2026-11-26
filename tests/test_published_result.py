"""Tests of the result lookup in the record layout."""

import pytest

from src.io import run_workflow as rw


def test_published_result_finds_the_single_result(tmp_path, monkeypatch):
    out = tmp_path / 'iteration2' / 'D0' / 'output'
    out.mkdir(parents=True)
    (out / 'TestFuelCellSystem_res.mat').write_bytes(b'')
    monkeypatch.setenv('FCS_SIMULATION_ROOT', str(tmp_path))

    assert rw.published_result('D0', 'iteration2') == out / 'TestFuelCellSystem_res.mat'


def test_published_result_rejects_unknown_set(tmp_path, monkeypatch):
    monkeypatch.setenv('FCS_SIMULATION_ROOT', str(tmp_path))
    with pytest.raises(ValueError):
        rw.published_result('D0', 'no_such_set')


def test_published_result_requires_exactly_one_file(tmp_path, monkeypatch):
    monkeypatch.setenv('FCS_SIMULATION_ROOT', str(tmp_path))
    with pytest.raises(FileNotFoundError):
        rw.published_result('T01', 'iteration1')
