import numpy as np
import pytest

from aspire.abinitio import CommonlineLUD
from aspire.nufft import backend_available
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
    pytest.param(np.float64, marks=pytest.mark.expensive),
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
        vols=AsymmetricVolume(L=resolution, C=1, K=100, seed=10).generate(),
        offsets=offsets,
        amplitudes=1,
        seed=0,
    )

    # Increase max_shift and set shift_step to be sub-pixel when using
    # random offsets in the Simulation. This improves common-line detection.
    max_shift = 0.20
    shift_step = 0.25

    # Set max_shift 1 pixel and shift_step to 1 pixel when using 0 offsets.
    if np.all(src.offsets == 0.0):
        max_shift = 1 / src.L
        shift_step = 1

    orient_est = CommonlineLUD(
        src,
        alpha=alpha,
        max_shift=max_shift,
        shift_step=shift_step,
        mask=False,
    )

    return src, orient_est


def test_estimate_rotations(src_orient_est_fixture):
    src, orient_est = src_orient_est_fixture

    if backend_available("cufinufft") and src.dtype == np.float32:
        pytest.skip("CI on GPU fails for singles.")

    orient_est.estimate_rotations()

    # Register estimates to ground truth rotations and compute the
    # angular distance between them (in degrees).
    # Assert that mean aligned angular distance is less than 3 degrees.
    mean_aligned_angular_distance(orient_est.rotations, src.rotations, degree_tol=3)
