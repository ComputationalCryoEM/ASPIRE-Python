import numpy as np
import pytest
from scipy import sparse

from aspire.abinitio import Orient3D
from aspire.source import Simulation
from aspire.volume import (
    AsymmetricVolume,
    CnSymmetricVolume,
    DnSymmetricVolume,
    OSymmetricVolume,
    TSymmetricVolume,
)

DTYPES = [np.float64]
RES = [89]
SYMMETRIES = [
    None,
    "C2",
    "C3",
    "C4",
    "C5",
    "C6",
    "D2",
    "D3",
    "D4",
    "D5",
    "D6",
    "D7",
    "T",
    "O",
]
N_IMGS = 100
SEED = 1980


@pytest.fixture(params=RES, ids=lambda x: f"resolution={x}", scope="module")
def resolution(request):
    return request.param


@pytest.fixture(params=SYMMETRIES, ids=lambda x: f"symmetry={x}", scope="module")
def symmetry(request):
    return request.param


@pytest.fixture(params=DTYPES, ids=lambda x: f"dtype={x}", scope="module")
def dtype(request):
    return request.param


@pytest.fixture(scope="module")
def volume(resolution, symmetry, dtype):
    if symmetry is None:
        return AsymmetricVolume(
            L=resolution, C=1, K=25, dtype=dtype, seed=SEED
        ).generate()

    if symmetry.startswith("C"):
        order = int(symmetry[1:])
        return CnSymmetricVolume(
            L=resolution, C=1, order=order, K=25, dtype=dtype, seed=SEED
        ).generate()

    if symmetry.startswith("D"):
        order = int(symmetry[1:])
        return DnSymmetricVolume(
            L=resolution, C=1, order=order, K=25, dtype=dtype, seed=SEED
        ).generate()

    if symmetry == "T":
        return TSymmetricVolume(
            L=resolution, C=1, K=25, dtype=dtype, seed=SEED
        ).generate()

    if symmetry == "O":
        return OSymmetricVolume(
            L=resolution, C=1, K=25, dtype=dtype, seed=SEED
        ).generate()


@pytest.fixture(scope="module")
def estimator(volume):
    """
    Build a simulated source and use ground-truth rotations so this test isolates
    shift-equation construction and shift recovery from orientation estimation error.
    """
    offset_scale = 1.5  # standard deviation of shifts
    offsets = np.random.normal(scale=offset_scale, size=(N_IMGS, 2))

    src = Simulation(
        n=N_IMGS,
        vols=volume,
        amplitudes=1,
        offsets=offsets,
        seed=SEED,
    ).cache()

    orient_est = Orient3D(src)
    orient_est.rotations = src.rotations

    return orient_est


def test_estimate_shifts(estimator):
    """
    Compare estimated shifts to ground truth after removing the nullspace of
    the shift equation matrix. See the following publication for more info on
    measuring shift estimation error:

    Y. Shkolnisky and A. Singer,
    Center of Mass Operators for Cryo-EM - Theory and Implementation,
    Modeling Nanoscale Imaging in Electron Microscopy,
    T. Vogt, W. Dahmen, and P. Binev (Eds.)
    Nanostructure Science and Technology Series,
    Springer, 2012, pp. 147–177
    """
    # Build the sparse common-line shift system Ax = b and solve it directly,
    # matching the solver used by estimate_shifts().
    A, b = estimator._get_shift_equations()
    lsqr_result = sparse.linalg.lsqr(A, b, atol=1e-8, btol=1e-8, iter_lim=100)
    x_est = lsqr_result[0]

    # Convert Simulation offsets to the internal LSQR convention:
    # estimate_shifts returns -x_est.reshape(n, 2)[:, ::-1].
    x_ref_internal = (-estimator.src.offsets[:, ::-1]).reshape(-1)

    # Use the SVD to separate the constrained directions from the nullspace,
    # which corresponds to global 3D translation ambiguity.
    _, s, Vt = np.linalg.svd(A.toarray(), full_matrices=False)

    # Estimate the effective rank of A and keep the constrained directions.
    sv_tol = 1e-2
    rank = int(np.sum(s > sv_tol * s[0]))
    V_nonnull = Vt[:rank].T

    # Compute relative error after projecting out the nullspace.
    num = np.linalg.norm(V_nonnull.T @ (x_ref_internal - x_est))
    den = np.linalg.norm(V_nonnull.T @ x_ref_internal)
    projected_rel_err = num / den

    # Check the shift error is within 15% of the reference shift norm.
    np.testing.assert_array_less(projected_rel_err, 0.15)

    # The projected relative error follows the legacy diagnostic, but it is not
    # a pixel-scale quantity. Below we check the same solution after aligning away
    # the nullspace component so the error is easier to interpret.
    V_null = Vt[rank:].T

    # Add the nullspace component to the estimate before comparing directly
    # against the reference shifts.
    x_err = x_ref_internal - x_est
    x_est_aligned = x_est + V_null @ (V_null.T @ x_err)

    # Convert back to ASPIRE shift convention and compute per-image Euclidean
    # shift error in pixels.
    est_shifts_aligned = -x_est_aligned.reshape(estimator.src.n, 2)[:, ::-1]
    per_img_err = np.linalg.norm(estimator.src.offsets - est_shifts_aligned, axis=1)
    mean_aligned_px = per_img_err.mean()

    # Check that aligned estimate errors are within 0.25 pixels on average.
    np.testing.assert_array_less(mean_aligned_px, 0.25)
