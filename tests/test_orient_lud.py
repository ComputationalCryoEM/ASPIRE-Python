import numpy as np
import pytest

from aspire.abinitio import CommonlineLUD
from aspire.source import Simulation
from aspire.utils import mean_aligned_angular_distance
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
    np.float64,
]

SPECTRAL_NORM_CONSTRAINT = [
    2 / 3,
    pytest.param(None, marks=pytest.mark.expensive),
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


@pytest.fixture(params=SPECTRAL_NORM_CONSTRAINT, ids=lambda x: f"alpha={x}")
def alpha(request):
    return request.param


@pytest.fixture
def src_orient_est_fixture(resolution, offsets, dtype, alpha):
    """Fixture for simulation source and orientation estimation object."""
    src = Simulation(
        n=60,
        L=resolution,
        vols=AsymmetricVolume(
            L=resolution, C=1, K=100, seed=10, dtype=dtype
        ).generate(),
        offsets=offsets,
        amplitudes=1,
        seed=0,
        dtype=dtype,
    )

    # Cache source to prevent regenerating images.
    src = src.cache()

    # Generate LUD orientation estimation object.
    orient_est = CommonlineLUD(
        src,
        alpha=alpha,
        mask=False,
        tol=0.005,  # Improves test speed
    )

    return src, orient_est


def test_estimate_rotations(src_orient_est_fixture):
    src, orient_est = src_orient_est_fixture

    # Estimate rotations
    est_rots = orient_est.estimate_rotations()

    # Register estimates to ground truth rotations and compute the
    # angular distance between them (in degrees).
    # Assert that mean aligned angular distance is less than 3 degrees.
    tol = 3

    # Using LUD without spectral norm constraint, ie. alpha=None,
    # on shifted images reduces estimated rotations accuracy.
    # This can be improved by using subpixel shift_step in CommonlineLUD.
    if orient_est.alpha is None and src.offsets.all() != 0:
        tol = 9
    mean_aligned_angular_distance(est_rots, src.rotations, degree_tol=tol)

    # Check dtype pass-through
    np.testing.assert_equal(src.dtype, est_rots.dtype)
