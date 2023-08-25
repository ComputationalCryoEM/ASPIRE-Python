import numpy as np
import pytest

from aspire.abinitio import CommonlineSDP
from aspire.nufft import backend_available
from aspire.source import Simulation
from aspire.utils import (
    Rotation,
    get_aligned_rotations,
    register_rotations,
    rots_to_clmatrix,
)
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
    np.float64,  # Temporary for gpu testing.
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
def src_orient_est_fixture(resolution, offsets, dtype):
    """Fixture for simulation source and orientation estimation object."""
    src = Simulation(
        n=50,
        L=resolution,
        vols=AsymmetricVolume(L=resolution, C=1, K=100, seed=0).generate(),
        offsets=offsets,
        amplitudes=1,
        seed=0,
    )

    max_shift = 1 / src.L  # sets max_shift to 1 pixel for 0 offsets.
    shift_step = 1
    if src.offsets.all() != 0:
        # These parameters improve common-line detection.
        max_shift = 0.20
        shift_step = 0.25  # Accounts for non-integer offsets.

    orient_est = CommonlineSDP(src, max_shift=max_shift, shift_step=shift_step)

    return src, orient_est


def test_estimate_rotations(src_orient_est_fixture):
    src, orient_est = src_orient_est_fixture

    if backend_available("cufinufft") and src.dtype == np.float32:
        pytest.skip("CI on gpu fails for singles.")

    orient_est.estimate_rotations()

    # Register estimates to ground truth rotations and compute the
    # angular distance between them (in degrees).
    Q_mat, flag = register_rotations(orient_est.rotations, src.rotations)
    regrot = get_aligned_rotations(orient_est.rotations, Q_mat, flag)
    mean_ang_dist = Rotation.mean_angular_distance(regrot, src.rotations) * 180 / np.pi

    # Assert that mean angular distance is less than 1 degrees.
    assert mean_ang_dist < 1


def test_construct_S(src_orient_est_fixture):
    src, orient_est = src_orient_est_fixture
    gt_cl_matrix = rots_to_clmatrix(src.rotations, orient_est.n_theta)
    S = orient_est._construct_S(gt_cl_matrix)

    # Check that S is symmetric
    assert np.allclose(S, S.T)

    # For uniformly distributed rotations the top eigenvalue should have multiplicity 3.
    # As such, we can expect that the top 3 eigenvlaues will all be close in value to their mean.
    eigs = np.linalg.eigvalsh(S)
    eigs = eigs[:3]
    eigs_mean = np.mean(eigs)

    # Check that the top 3 eigenvalues are all within 10% of the their mean.
    assert (abs((eigs - eigs_mean) / eigs_mean) < 0.10).all()
