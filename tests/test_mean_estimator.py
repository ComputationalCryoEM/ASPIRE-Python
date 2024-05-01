import os.path
import tempfile

import numpy as np
import pytest
from pytest import raises

from aspire.basis import Coef, FBBasis3D
from aspire.operators import RadialCTFFilter
from aspire.reconstruction import MeanEstimator
from aspire.source.simulation import Simulation
from aspire.utils import grid_3d

DATA_DIR = os.path.join(os.path.dirname(__file__), "saved_test_data")

# Params

SEED = 1616

DTYPE = [np.float32, np.float64]
L = [
    8,
    9,
]

PRECONDITIONERS = ["none", None]  # default, circulant

# Fixtures.


@pytest.fixture(params=L, ids=lambda x: f"L={x}", scope="module")
def L(request):
    return request.param


@pytest.fixture(params=DTYPE, ids=lambda x: f"dtype={x}", scope="module")
def dtype(request):
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


@pytest.fixture(
    params=PRECONDITIONERS, ids=lambda x: f"preconditioner={x}", scope="module"
)
def estimator(request, sim, basis):
    preconditioner = request.param
    return MeanEstimator(sim, basis=basis, preconditioner=preconditioner)


@pytest.fixture(scope="module")
def mask(L):
    return grid_3d(L)["r"] < 1


# Tests
def test_estimate_resolution_error(sim, basis):
    """
    Test mismatched resolutions yields a relevant error message.
    """

    with raises(ValueError, match=r".*resolution.*"):
        # This basis is intentionally the wrong resolution.
        incorrect_basis = FBBasis3D(sim.L + 1, dtype=sim.dtype)

        _ = MeanEstimator(sim, basis=incorrect_basis, preconditioner="none")


def test_estimate(sim, estimator, mask):
    estimate = estimator.estimate()

    est = estimate.asnumpy() * mask
    vol = sim.vols.asnumpy() * mask

    np.testing.assert_allclose(
        est / np.linalg.norm(est), vol / np.linalg.norm(vol), atol=0.1
    )


def test_adjoint(sim, basis, estimator, mask):
    # Mean coefs formed by backprojections
    mean_b_coef = estimator.src_backward()

    # Evaluate mean coefs into a volume
    est = Coef(basis, mean_b_coef).evaluate() * mask

    # Mask off corners of volume
    vol = sim.vols.asnumpy() * mask

    # Assert the mean volume is close to original volume
    np.testing.assert_allclose(
        est / np.linalg.norm(est), vol / np.linalg.norm(vol), atol=0.11
    )


def test_checkpoint(sim, basis, estimator):
    """Exercise the checkpointing and max iterations branches."""
    test_iter = 2
    with tempfile.TemporaryDirectory() as tmp_input_dir:
        prefix = os.path.join(tmp_input_dir, "new", "dirs", "chk")
        _estimator = MeanEstimator(
            sim,
            basis=basis,
            preconditioner="none",
            checkpoint_iterations=test_iter,
            maxiter=test_iter + 1,
            checkpoint_prefix=prefix,
        )

        # Assert we raise when reading `maxiter`.
        with raises(RuntimeError, match="Unable to converge!"):
            _ = _estimator.estimate()

        # Load the checkpoint coefficients while tmp_input_dir exists.
        b_chk = np.load(f"{prefix}_iter{test_iter:04d}.npy")

        # Restart estimate from checkpoint
        _ = estimator.estimate(b_coef=b_chk)


def test_checkpoint_args(sim, basis):
    with tempfile.TemporaryDirectory() as tmp_input_dir:
        prefix = os.path.join(tmp_input_dir, "chk")

        for junk in [-1, 0, "abc"]:
            # Junk `checkpoint_iterations` values
            with raises(
                ValueError, match=r".*iterations.*should be a positive integer.*"
            ):
                _ = MeanEstimator(
                    sim,
                    basis=basis,
                    preconditioner="none",
                    checkpoint_iterations=junk,
                    checkpoint_prefix=prefix,
                )
            # Junk `maxiter` values
            with raises(ValueError, match=r".*maxiter.*should be a positive integer.*"):
                _ = MeanEstimator(
                    sim,
                    basis=basis,
                    preconditioner="none",
                    maxiter=junk,
                    checkpoint_prefix=prefix,
                )
