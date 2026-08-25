import numpy as np
import pytest

from aspire.abinitio import CommonlineNUG, compare_rots_sym, g_sync
from aspire.downloader import emdb_2660
from aspire.source import Simulation
from aspire.utils import mean_aligned_angular_distance
from aspire.volume import CnSymmetricVolume, DnSymmetricVolume, TSymmetricVolume

DTYPE = [np.float64, np.float32]
RESOLUTION = [48, 49]
N_IMG = [30]
OFFSETS = [0, None]
SYMMETRY = ["C1", "C3", "C4", "D3", "D4"]
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


@pytest.fixture(params=SYMMETRY, ids=lambda x: f"symmetry={x}", scope="module")
def symmetry(request):
    return request.param


############
# Fixtures #
############


@pytest.fixture(scope="module")
def volume(symmetry, resolution, dtype):
    if symmetry == "C1":
        vol = emdb_2660().astype(dtype).downsample(resolution)
    if symmetry == "C3":
        vol = CnSymmetricVolume(
            L=resolution, order=3, C=1, K=100, dtype=dtype, seed=SEED
        ).generate()
    if symmetry == "C4":
        vol = CnSymmetricVolume(
            L=resolution, order=4, C=1, K=100, dtype=dtype, seed=SEED
        ).generate()
    if symmetry == "D3":
        vol = DnSymmetricVolume(
            L=resolution, order=3, C=1, K=100, dtype=dtype, seed=SEED
        ).generate()
    if symmetry == "D4":
        vol = DnSymmetricVolume(
            L=resolution, order=4, C=1, K=100, dtype=dtype, seed=SEED
        ).generate()
    return vol


@pytest.fixture(scope="module")
def source(n_img, offsets, volume):
    src = Simulation(
        n=n_img,
        vols=volume,
        offsets=offsets,
        amplitudes=1,
        seed=SEED,
    ).cache()  # Precompute image stack
    return src


@pytest.fixture(scope="module")
def orient_est(source):
    max_shift = 0
    shift_step = 1
    if source.offsets.all() != 0:
        max_shift = 0.30
        shift_step = 0.5
    orient_est = CommonlineNUG(
        source,
        max_shift=max_shift,
        shift_step=shift_step,
        max_iter=201,
        pr_iters=None,
        verbose=False,
    )
    orient_est.estimate_rotations()
    return orient_est


#########
# Tests #
#########


def test_smoke(dtype, Volume):
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
        pr_iters=1,
        S2_grid=50,
        max_shift=0,
        verbose=True,
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
    """
    Check mean squared error between estimates and ground truth
    pairwise rotations, Rij. This serves as a reference to the error
    metric used by the researcher in the related publication.
    """
    MSE = compare_rots_sym(
        orient_est.rotations, orient_est.src.rotations, orient_est.sym_grp
    )
    np.testing.assert_array_less(MSE, 0.15)


@pytest.mark.expensive
def test_estimate_rotations(orient_est):
    """
    Check that the mean angular distance between estimates and ground
    truth, after symmetry synchronization and global alignment, are
    within 10 degrees.
    """
    gt_rots = orient_est.src.rotations
    if orient_est.sym_grp.order > 1:
        gt_rots = g_sync(
            orient_est.rotations, orient_est.src.rotations, orient_est.sym_grp
        )
    mean_aligned_angular_distance(orient_est.rotations, gt_rots, 10.0)


def test_unspupported_symmetry_raises(dtype):
    """
    Check that we raise for symmetries other than Cn/Dn.
    """
    vol = TSymmetricVolume(L=16, C=1, K=10, dtype=dtype).generate()
    src = Simulation(n=3, vols=vol)

    with pytest.raises(
        ValueError, match="supports cyclic, dihedral, and asymmetric molecules"
    ):
        _ = CommonlineNUG(src)


def test_symmetry_logging(caplog):
    """
    Check expected log messages for mismatch or unprovided symmetry.
    """
    vol = CnSymmetricVolume(L=16, order=3, C=1, dtype=np.float64, seed=SEED).generate()
    src = Simulation(n=3, vols=vol).cache()

    # Check unprovided symmetry populates with source symmetry and logs messsage
    caplog.clear()
    with caplog.at_level("INFO"):
        orient_est = CommonlineNUG(src)

    assert str(orient_est.sym_grp) == "C3"
    assert "Symmetry not provided. Using Source symmetry: C3" in caplog.text

    # Check we use provided symmetry on mismatch, with log
    caplog.clear()
    with caplog.at_level("INFO"):
        orient_est = CommonlineNUG(src, symmetry="D3")

    assert str(orient_est.sym_grp) == "D3"
    assert "Provided symmetry, D3, does not match source, C3" in caplog.text
    assert "Using provided symmetry: D3" in caplog.text
