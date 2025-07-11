import numpy as np
import pytest

from aspire.abinitio import CommonlineIRLS
from aspire.source import Simulation
from aspire.utils import mean_aligned_angular_distance
from aspire.volume import AsymmetricVolume

RESOLUTION = [
    32,
    pytest.param(33, marks=pytest.mark.expensive),
]

OFFSETS = [
    0,
    None,  # Defaults to random offsets.
]

DTYPES = [
    np.float32,
    pytest.param(np.float64, marks=pytest.mark.expensive),
]

SPECTRAL_NORM_CONSTRAINT = [
    None,
    2 / 3,
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
    orient_est = CommonlineIRLS(
        source,
        alpha=alpha,
        adp_proj=adp_proj,
        delta_mu_l=0.4,  # Ensures branch is tested
        mask=False,
        tol=0.005,  # Improves test speed
        num_itrs=3,  # Improves test speed
    )

    return orient_est


def test_estimate_rotations(source, orient_est):
    # Estimate rotations
    est_rots = orient_est.estimate_rotations()

    # Register estimates to ground truth rotations and compute the
    # angular distance between them (in degrees).
    # Assert that mean aligned angular distance is less than 5 degrees.
    tol = 5
    mean_aligned_angular_distance(est_rots, source.rotations, degree_tol=tol)

    # Check dtype pass-through
    np.testing.assert_equal(source.dtype, est_rots.dtype)


def test_adjoint_property_A(dtype):
    """
    Test <A u, v> = <u, AT v> for random symmetric matrix `u` and
    random vector `v`.
    """
    n = 10
    u = np.random.rand(2 * n, 2 * n).astype(dtype, copy=False)
    u = (u + u.T) / 2
    v = np.random.rand(3 * n).astype(dtype, copy=False)

    Au = CommonlineIRLS._compute_AX(u)
    ATv = CommonlineIRLS._compute_ATy(v)

    lhs = np.dot(Au, v)
    rhs = np.dot(u.flatten(), ATv.flatten())

    np.testing.assert_allclose(lhs, rhs, rtol=1e-05, atol=1e-08)
