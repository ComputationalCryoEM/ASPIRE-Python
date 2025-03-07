import os.path
import tempfile

import numpy as np
import pytest
from pytest import raises

from aspire.basis import Coef, FBBasis3D
from aspire.operators import RadialCTFFilter
from aspire.reconstruction import WeightedVolumesEstimator
from aspire.source.simulation import Simulation
from aspire.utils import grid_3d

DATA_DIR = os.path.join(os.path.dirname(__file__), "saved_test_data")

# Params

SEED = 1617

DTYPE = [np.float32, np.float64]
L = [
    8,
    9,
]

PRECONDITIONERS = [
    None,
    "circulant",
    pytest.param("none", marks=pytest.mark.expensive),
    pytest.param("", marks=pytest.mark.expensive),
]

# Fixtures.


@pytest.fixture(params=L, ids=lambda x: f"L={x}", scope="module")
def L(request):
    return request.param


@pytest.fixture(params=DTYPE, ids=lambda x: f"dtype={x}", scope="module")
def dtype(request):
    return request.param


@pytest.fixture(params=DTYPE, ids=lambda x: f"weights_dtype={x}", scope="module")
def weights_dtype(request):
    return request.param


@pytest.fixture(scope="module")
def sim(L, dtype):
    sim = Simulation(
        L=L,
        n=256,
        C=1,  # single volume
        unique_filters=[
            RadialCTFFilter(defocus=d) for d in np.linspace(1.5e4, 2.5e4, 7)
        ],
        dtype=dtype,
        seed=SEED,
    )

    sim = sim.cache()  # precompute images

    return sim


@pytest.fixture(scope="module")
def basis(L, dtype):
    return FBBasis3D(L, dtype=dtype)


@pytest.fixture(scope="module")
def weights(sim, weights_dtype):
    # Construct simple test weights;
    # one uniform positive and negative weighted volume respectively.
    r = 2  # Number of weighted volumes
    weights = np.ones((sim.n, r), dtype=weights_dtype) / np.sqrt(
        sim.n, dtype=weights_dtype
    )
    weights[:, 1] *= -1  # negate second weight vector

    return weights


@pytest.fixture(
    params=PRECONDITIONERS, ids=lambda x: f"preconditioner={x}", scope="module"
)
def estimator(request, sim, basis, weights):
    preconditioner = request.param

    return WeightedVolumesEstimator(
        weights, sim, basis=basis, preconditioner=preconditioner
    )


@pytest.fixture(scope="module")
def mask(L):
    return grid_3d(L)["r"] < 1


# Tests
def test_resolution_error(sim, basis, weights):
    """
    Test mismatched resolutions yields a relevant error message.
    """

    with raises(ValueError, match=r".*resolution.*"):
        # This basis is intentionally the wrong resolution.
        incorrect_basis = FBBasis3D(sim.L + 1, dtype=sim.dtype)

        _ = WeightedVolumesEstimator(
            weights, sim, basis=incorrect_basis, preconditioner="none"
        )


def test_estimate(sim, estimator, mask):
    estimate = estimator.estimate()

    est = estimate * mask
    vol = sim.vols * mask

    for i, w in enumerate([1, -1]):
        np.testing.assert_allclose(
            w * est[i] / np.linalg.norm(est[i]), vol / np.linalg.norm(vol), atol=0.1
        )


def test_src_adjoint(sim, basis, estimator):
    """
    Test the built-in source based estimator's `src_backward` has
    adjoint like relationship with simulated image generation.
    """

    v = sim.vols.asnumpy()[0]  # random volume
    proj = sim.images[:]  # projections of v
    u = proj.asnumpy()  # u = proj

    # `src_backward` scales by 1/n
    backproj = Coef(basis, estimator.src_backward() * sim.n).evaluate()

    lhs = np.dot(proj.asnumpy().flatten(), u.flatten())

    for i, w in enumerate([1, -1]):
        rhs = np.dot(backproj[i].asnumpy().flatten(), w * v.flatten())
        np.testing.assert_allclose(lhs, rhs, rtol=0.02)


def test_checkpoint(sim, basis, estimator, weights):
    """Exercise the checkpointing and max iterations branches."""
    test_iter = 2
    with tempfile.TemporaryDirectory() as tmp_input_dir:
        prefix = os.path.join(tmp_input_dir, "new", "dirs", "chk")
        _estimator = WeightedVolumesEstimator(
            weights,
            sim,
            basis=basis,
            preconditioner="none",
            checkpoint_iterations=test_iter,
            maxiter=test_iter + 1,
            checkpoint_prefix=prefix,
        )

        _ = _estimator.estimate()

        # Load the checkpoint coefficients while tmp_input_dir exists.
        x_chk = np.load(f"{prefix}_iter{test_iter:04d}.npy")

        # Restart estimate from checkpoint
        _ = estimator.estimate(x0=x_chk)


def test_checkpoint_args(sim, basis, weights):
    with tempfile.TemporaryDirectory() as tmp_input_dir:
        prefix = os.path.join(tmp_input_dir, "chk")

        for junk in [-1, 0, "abc"]:
            # Junk `checkpoint_iterations` values
            with raises(
                ValueError, match=r".*iterations.*should be a positive integer.*"
            ):
                _ = WeightedVolumesEstimator(
                    weights,
                    sim,
                    basis=basis,
                    preconditioner="none",
                    checkpoint_iterations=junk,
                    checkpoint_prefix=prefix,
                )
            # Junk `maxiter` values
            with raises(ValueError, match=r".*maxiter.*should be a positive integer.*"):
                _ = WeightedVolumesEstimator(
                    weights,
                    sim,
                    basis=basis,
                    preconditioner="none",
                    maxiter=junk,
                    checkpoint_prefix=prefix,
                )
