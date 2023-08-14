import numpy as np
import pytest

from aspire.abinitio import CommonlineSDP
from aspire.source import Simulation
from aspire.utils import Rotation, get_aligned_rotations, register_rotations
from aspire.volume import AsymmetricVolume

RESOLUTION = [
    32,
    33,
]

OFFSETS = [
    None,  # Defaults to random offsets.
    0,
]

DTYPES = [
    np.float32,
    pytest.param(np.float64, marks=pytest.mark.expensive),
]


@pytest.fixture(params=RESOLUTION, ids=lambda x: f"resolution={x}")
def resolution(request):
    return request.param


@pytest.fixture(params=OFFSETS, ids=lambda x: f"offsets={x}")
def offsets(request):
    return request.param


@pytest.fixture(params=DTYPES, ids=lambda x: f"dtype={x}")
def dtype(request):
    return request.param


@pytest.fixture
def simulation_fixture(resolution, offsets, dtype):
    src = Simulation(
        n=50,
        L=resolution,
        vols=AsymmetricVolume(L=resolution, C=1, K=100).generate(),
        offsets=offsets,
        amplitudes=1,
    )

    return src


def test_estimate_rotations(simulation_fixture):
    src = simulation_fixture
    orient_est = CommonlineSDP(src)

    orient_est.estimate_rotations()

    # Register estimates to ground truth rotations and compute the
    # angular distance between them (in degrees).
    Q_mat, flag = register_rotations(orient_est.rotations, src.rotations)
    regrot = get_aligned_rotations(orient_est.rotations, Q_mat, flag)
    ang_dist = np.zeros(src.n, dtype=src.dtype)
    for i in range(src.n):
        ang_dist[i] = (
            Rotation.angle_dist(
                regrot[i],
                src.rotations[i],
                dtype=src.dtype,
            )
            * 180
            / np.pi
        )

    # Assert that mean angular distance is less than 1 degree (10 degrees with shifts).
    degree_tol = 1
    if src.offsets.all() != 0:
        degree_tol = 10
    assert np.mean(ang_dist) < degree_tol
