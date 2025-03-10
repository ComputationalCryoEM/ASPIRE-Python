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
    pytest.param(np.float64, marks=pytest.mark.expensive),
]

SPECTRAL_NORM_CONSTRAINT = [
    2 / 3,
    pytest.param(None, marks=pytest.mark.expensive),
]

ADAPTIVE_PROJECTION = [
    True,
    False,
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


@pytest.fixture(params=ADAPTIVE_PROJECTION, ids=lambda x: f"adp_proj={x}")
def adp_proj(request):
    return request.param


@pytest.fixture
def source(resolution, offsets, dtype):
    """Fixture for simulation source object."""
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

    return src


@pytest.fixture
def orient_est(source, alpha, adp_proj):
    """Fixture for LUD orientation estimation object."""
    # Generate LUD orientation estimation object.
    orient_est = CommonlineLUD(
        source,
        alpha=alpha,
        adp_proj=adp_proj,
        delta_mu_l=0.4,  # Ensures branch is tested
        mask=False,
        tol=0.005,  # Improves test speed
    )

    return orient_est


def test_estimate_rotations(source, orient_est):
    # Estimate rotations
    est_rots = orient_est.estimate_rotations()

    # Register estimates to ground truth rotations and compute the
    # angular distance between them (in degrees).
    # Assert that mean aligned angular distance is less than 3 degrees.
    tol = 3

    # Using LUD without spectral norm constraint, ie. alpha=None,
    # on shifted images reduces estimated rotations accuracy.
    # This can be improved by using subpixel shift_step in CommonlineLUD.
    if orient_est.alpha is None and source.offsets.all() != 0:
        tol = 9
    mean_aligned_angular_distance(est_rots, source.rotations, degree_tol=tol)

    # Check dtype pass-through
    np.testing.assert_equal(source.dtype, est_rots.dtype)
