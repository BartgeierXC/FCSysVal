import numpy as np
import pytest

from src.utils import om_util


class FakeOMCSession:
    def __init__(self, variables, data):
        self.variables = variables
        self.data = data

    def sendExpression(self, expression):
        if expression.startswith("readSimulationResultVars"):
            return self.variables
        if expression.startswith("readSimulationResult("):
            return self.data
        return None


def mock_mat_result(monkeypatch, times, values):
    monkeypatch.setattr(
        om_util,
        "OMCSessionZMQ",
        lambda: FakeOMCSession(["time", "value"], [times, values]),
    )


def test_import_mat_result_matches_nearest_time_and_snaps_to_grid(monkeypatch):
    mock_mat_result(
        monkeypatch,
        [0.0, 1.00000002, 1.0008, 2.0],
        [0.0, 10.0, 99.0, 20.0],
    )

    result = om_util.import_mat_result("result.mat")

    np.testing.assert_array_equal(result["time"], [0.0, 1.0, 2.0])
    np.testing.assert_array_equal(result["value"], [0.0, 10.0, 20.0])


def test_import_mat_result_omits_grid_time_without_nearby_sample(monkeypatch):
    mock_mat_result(monkeypatch, [0.0, 0.25, 2.0], [0.0, 5.0, 20.0])

    result = om_util.import_mat_result("result.mat")

    np.testing.assert_array_equal(result["time"], [0.0, 2.0])
    np.testing.assert_array_equal(result["value"], [0.0, 20.0])


def test_import_mat_result_scales_tolerance_with_timestep(monkeypatch):
    mock_mat_result(
        monkeypatch,
        [0.0, 2.001, 2.0018, 4.0],
        [0.0, 10.0, 99.0, 20.0],
    )

    result = om_util.import_mat_result("result.mat", timestep=2)

    np.testing.assert_array_equal(result["time"], [0.0, 2.0, 4.0])
    np.testing.assert_array_equal(result["value"], [0.0, 10.0, 20.0])


@pytest.mark.parametrize(
    ("timestep", "rtol"),
    [(0, 1e-3), (np.inf, 1e-3), (1, -1e-3), (1, np.nan)],
)
def test_import_mat_result_rejects_invalid_sampling_parameters(monkeypatch, timestep, rtol):
    mock_mat_result(monkeypatch, [0.0], [0.0])

    with pytest.raises(ValueError):
        om_util.import_mat_result("result.mat", timestep=timestep, rtol=rtol)
