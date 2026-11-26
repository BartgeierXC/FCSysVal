import pandas as pd
import pytest

from src.utils.val_doe import remove_twins


def test_remove_twins_merges_close_points_and_keeps_singles_by_default():
    df = pd.DataFrame({"a": [1.0, 1.0001, 5.0, 10.0]})
    result = remove_twins(df, dist=1e-3, remove_single=False)
    assert sorted(result["a"].tolist()) == pytest.approx([1.00005, 5.0, 10.0])


def test_remove_twins_removes_singles_when_remove_single_true():
    df = pd.DataFrame({"a": [1.0, 1.0001, 5.0, 10.0]})
    result = remove_twins(df, dist=1e-3, remove_single=True)
    # Only the merged pair (1.0, 1.0001) survives; the two isolated points are dropped.
    assert len(result) == 1
    assert result["a"].iloc[0] == pytest.approx(1.00005)


def test_remove_twins_default_matches_remove_single_true():
    df = pd.DataFrame({"a": [1.0, 1.0001, 5.0, 10.0]})
    assert remove_twins(df, dist=1e-3).equals(remove_twins(df, dist=1e-3, remove_single=True))


def test_remove_twins_all_singles_removed_returns_empty_dataframe():
    df = pd.DataFrame({"a": [1.0, 5.0, 10.0]})
    result = remove_twins(df, dist=1e-3, remove_single=True)
    assert result.empty
    assert list(result.columns) == ["a"]


def test_remove_twins_single_row_removed_when_remove_single_true():
    df = pd.DataFrame({"a": [1.0]})
    result = remove_twins(df, remove_single=True)
    assert result.empty


def test_remove_twins_single_row_kept_when_remove_single_false():
    df = pd.DataFrame({"a": [1.0]})
    result = remove_twins(df, remove_single=False)
    assert len(result) == 1


def test_remove_twins_empty_dataframe_raises():
    with pytest.raises(ValueError):
        remove_twins(pd.DataFrame({"a": []}))
