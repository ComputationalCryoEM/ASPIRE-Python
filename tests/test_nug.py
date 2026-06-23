import numpy as np
import pytest

from aspire.abinitio import CommonlineNUG, compare_rots_sym, g_sync
from aspire.source import Simulation
from aspire.utils import mean_aligned_angular_distance
from aspire.volume import CnSymmetricVolume, DnSymmetricVolume, TSymmetricVolume

DTYPE = [np.float64, np.float32]
RESOLUTION = [48, 49]
N_IMG = [15]
OFFSETS = [0, None]
ORDER = [3, 4]
PR = [False]
SEED = 1980
VOLUME = [
    CnSymmetricVolume,
    DnSymmetricVolume,
]


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


@pytest.fixture(params=VOLUME, ids=lambda x: f"Volume={x}", scope="module")
def Volume(request):
    return request.param


############
# Fixtures #
############


@pytest.fixture(scope="module")
def source(n_img, resolution, dtype, offsets, order, Volume):
    vol = Volume(
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
    max_shift = 0
    shift_step = 1
    if source.offsets.all() != 0:
        max_shift = 0.20
        shift_step = 0.25
    orient_est = CommonlineNUG(
        source,
        max_shift=max_shift,
        shift_step=shift_step,
        perform_pr=proximal_refine,
        verbose=False,
    )
    orient_est.estimate_rotations()
    return orient_est


#########
# Tests #
#########


def test_smoke_nug(dtype, Volume):
    """
    Perform quick smoke test since other tests are long running.
    """
    vol = Volume(L=32, order=3, C=1, dtype=dtype, seed=SEED).generate()

    src = Simulation(
        n=20,
        vols=vol,
        offsets=0,
        amplitudes=1,
        seed=SEED,
    ).cache()

    orient_est = CommonlineNUG(
        src,
        Lmax=4,
        T=5,
        max_iter=10,
        S2_grid=50,
        max_shift=0,
        mask=False,
        verbose=False,
    )

    rots = orient_est.estimate_rotations()

    assert rots.shape == (src.n, 3, 3)
    assert rots.dtype == dtype


@pytest.mark.expensive
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


@pytest.mark.expensive
def test_estimate_rotations_pairwise(orient_est):
    """ """
    MSE = compare_rots_sym(
        orient_est.rotations, orient_est.src.rotations, orient_est.sym_grp
    )
    np.testing.assert_array_less(MSE, 0.1)


@pytest.mark.expensive
def test_estimate_rotations(orient_est):
    gt_rots_synced = g_sync(
        orient_est.rotations, orient_est.src.rotations, orient_est.sym_grp
    )
    mean_aligned_angular_distance(orient_est.rotations, gt_rots_synced, 8.0)


def test_unspupported_symmetry_raises(dtype):
    vol = TSymmetricVolume(L=16, C=1, K=10, dtype=dtype).generate()
    src = Simulation(n=3, vols=vol)

    with pytest.raises(ValueError, match="supports cyclic or dihedral symmetry"):
        _ = CommonlineNUG(src)
