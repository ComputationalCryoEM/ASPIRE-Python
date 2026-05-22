import numpy as np
import pytest

from aspire.abinitio import CommonlineNUG
from aspire.source import Simulation
from aspire.volume import CnSymmetricVolume, SymmetryGroup

DTYPE = [np.float32, pytest.param(np.float64, marks=pytest.mark.expensive)]
RESOLUTION = [48, pytest.param(48, marks=pytest.mark.expensive)]
N_IMG = [5]
OFFSETS = [0]
ORDER = [3, pytest.param(4, marks=pytest.mark.expensive)]
PR = [False]
SEED = 1980


@pytest.fixture(params=DTYPE, ids=lambda x: f"dtype={x}", scope="module")
def dtype(request):
    return request.param


@pytest.fixture(params=RESOLUTION, ids=lambda x: f"resolution={x}", scope="module")
def resolution(request):
    return request.param


@pytest.fixture(params=N_IMG, ids=lambda x: f"n images={x}", scope="module")
def n_img(request):
    return request.param


@pytest.fixture(params=OFFSETS, ids=lambda x: f"offsets={x}", scope="module")
def offsets(request):
    return request.param


@pytest.fixture(params=ORDER, ids=lambda x: f"order={x}", scope="module")
def order(request):
    return request.param


@pytest.fixture(params=PR, ids=lambda x: f"proximal_refine={x}", scope="module")
def proximal_refine(request):
    return request.param


############
# Fixtures #
############


@pytest.fixture(scope="module")
def source(n_img, resolution, dtype, offsets, order):
    vol = CnSymmetricVolume(
        L=resolution, order=order, C=1, K=100, dtype=dtype, seed=SEED
    ).generate()

    src = Simulation(
        n=n_img,
        L=resolution,
        vols=vol,
        offsets=offsets,
        amplitudes=1,
        seed=SEED,
    )
    src = src.cache()  # Precompute image stack

    return src


@pytest.fixture(scope="module")
def orient_est(source, proximal_refine):
    orient_est = CommonlineNUG(
        source,
        max_shift=0,
        perform_pr=proximal_refine,
    )
    orient_est.estimate_rotations()
    return orient_est


#########
# Tests #
#########


def test_dtypes(orient_est):
    """
    Check dtypes for each major step of the algorithm.
    """
    assert orient_est.dtype == orient_est.src.dtype

    # Intermediate steps use doubles
    for Ci in orient_est.C:
        assert Ci.dtype == np.float64

    for Xi in orient_est.X_est:
        assert Xi.dtype == np.float64

    assert orient_est.rotations.dtype == orient_est.dtype


def test_estimate_rotations_pairwise(orient_est):
    """ """
    MSE = compare_rots_sym(
        orient_est.rotations, orient_est.src.rotations, orient_est.sym_grp
    )
    np.testing.assert_array_less(MSE, 0.3)


###########
# Helpers #
###########


def compare_rots_sym(R_est, R_true, sym):
    N = R_true.shape[0]
    sym_euler = SymmetryGroup.parse(sym).matrices
    order = sym_euler.shape[0]
    J = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
    error = np.zeros((N, N))
    errorJ = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            e = np.zeros(order)
            eJ = np.zeros(order)
            for s in range(order):
                Rs = sym_euler[s]
                e[s] = (
                    np.linalg.norm(R_est[i].T @ R_est[j] - R_true[i].T @ Rs @ R_true[j])
                    ** 2
                )
                eJ[s] = (
                    np.linalg.norm(
                        R_est[i].T @ R_est[j] - J @ R_true[i].T @ Rs @ R_true[j] @ J
                    )
                    ** 2
                )
            error[i, j] = e.min()
            errorJ[i, j] = eJ.min()
    E = min(error.sum(), errorJ.sum()) / N**2
    return E
