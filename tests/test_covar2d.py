import os
import os.path

import numpy as np
import pytest
from pytest import raises

from aspire.basis import FFBBasis2D
from aspire.covariance import RotCov2D
from aspire.noise import WhiteNoiseAdder
from aspire.operators import RadialCTFFilter
from aspire.source.simulation import _LegacySimulation
from aspire.utils import randi, utest_tolerance
from aspire.volume import Volume

DATA_DIR = os.path.join(os.path.dirname(__file__), "saved_test_data")


IMG_SIZES = [8]
DTYPES = [np.float32]
# Basis used in FSPCA for class averaging.
BASIS = [
    FFBBasis2D,
]

# Hard coded to match legacy files.
NOISE_VAR = 1.3957e-4

# Cover `test_shrinkage`
SHRINKERS = [None, "frobenius_norm", "operator_norm", "soft_threshold"]

CTF_ENABLED = [True, False]


@pytest.fixture(params=CTF_ENABLED, ids=lambda x: f"ctf={x}")
def ctf_enabled(request):
    return request.param


@pytest.fixture(params=SHRINKERS, ids=lambda x: f"shrinker={x}")
def shrinker(request):
    return request.param


@pytest.fixture(params=DTYPES, ids=lambda x: f"dtype={x}")
def dtype(request):
    return request.param


@pytest.fixture(params=IMG_SIZES, ids=lambda x: f"img_size={x}")
def img_size(request):
    return request.param


@pytest.fixture
def volume(dtype, img_size):
    # Get a volume
    v = Volume(
        np.load(os.path.join(DATA_DIR, "clean70SRibosome_vol_down8.npy")).astype(dtype),
        pixel_size=5.0 * 65 / 8,
    )
    # 1e3 is hardcoded to match legacy test files.
    return v * 1.0e3


@pytest.fixture(params=BASIS, ids=lambda x: f"basis={x}")
def basis(request, img_size, dtype):
    cls = request.param
    # Setup a Basis
    basis = cls(img_size, dtype=dtype)
    return basis


@pytest.fixture
def cov2d_fixture(volume, basis, ctf_enabled):
    """
    Cov2D Test Fixture.
    """
    n = 32

    # Default CTF params
    unique_filters = None
    h_idx = None
    h_ctf_fb = None
    # Popluate CTF
    if ctf_enabled:
        unique_filters = [
            RadialCTFFilter(200, defocus=d, Cs=2.0, alpha=0.1)
            for d in np.linspace(1.5e4, 2.5e4, 7)
        ]

        # Copied from simulation defaults to match legacy test files.
        h_idx = randi(len(unique_filters), n, seed=0) - 1

        h_ctf_fb = [
            basis.filter_to_basis_mat(f, pixel_size=volume.pixel_size)
            for f in unique_filters
        ]

    noise_adder = WhiteNoiseAdder(var=NOISE_VAR)

    sim = _LegacySimulation(
        n=n,
        vols=volume,
        unique_filters=unique_filters,
        filter_indices=h_idx,
        offsets=0.0,
        amplitudes=1.0,
        dtype=volume.dtype,
        noise_adder=noise_adder,
    )
    sim = sim.cache()

    cov2d = RotCov2D(basis)
    coef_clean = basis.evaluate_t(sim.projections[:])
    coef = basis.evaluate_t(sim.images[:])

    return sim, cov2d, coef_clean, coef, h_ctf_fb, h_idx


def test_get_mean(cov2d_fixture):
    results = np.load(os.path.join(DATA_DIR, "clean70SRibosome_cov2d_mean.npy"))
    cov2d, coef_clean = cov2d_fixture[1], cov2d_fixture[2]

    mean_coef = cov2d._get_mean(coef_clean.asnumpy())
    np.testing.assert_allclose(results, mean_coef, atol=utest_tolerance(cov2d.dtype))


def test_get_covar(cov2d_fixture):
    results = np.load(
        os.path.join(DATA_DIR, "clean70SRibosome_cov2d_covar.npz"),
    )

    cov2d, coef_clean = cov2d_fixture[1], cov2d_fixture[2]

    covar_coef = cov2d._get_covar(coef_clean.asnumpy())

    for im, mat in enumerate(results.values()):
        np.testing.assert_allclose(mat, covar_coef[im], rtol=1e-05)


def test_get_mean_ctf(cov2d_fixture, ctf_enabled):
    """
    Compare `get_mean` (no CTF args) with `_get_mean` (no CTF model).
    """
    sim, cov2d, coef_clean, coef, h_ctf_fb, h_idx = cov2d_fixture

    mean_coef_ctf = cov2d.get_mean(coef, h_ctf_fb, h_idx)

    tol = utest_tolerance(sim.dtype)
    if ctf_enabled:
        result = np.load(os.path.join(DATA_DIR, "clean70SRibosome_cov2d_meanctf.npy"))
    else:
        result = cov2d._get_mean(coef_clean.asnumpy())
        tol = 0.002

    np.testing.assert_allclose(mean_coef_ctf.asnumpy()[0], result, atol=tol)


def test_get_cwf_coefs_clean(cov2d_fixture):
    results = np.load(
        os.path.join(DATA_DIR, "clean70SRibosome_cov2d_cwf_coef_clean.npy")
    )

    cov2d, coef_clean = cov2d_fixture[1], cov2d_fixture[2]

    coef_cwf_clean = cov2d.get_cwf_coefs(coef_clean, noise_var=0)
    np.testing.assert_allclose(
        results, coef_cwf_clean, atol=utest_tolerance(cov2d.dtype)
    )


def test_get_cwf_coefs_clean_ctf(cov2d_fixture):
    """
    Test case of clean images (coef_clean and noise_var=0)
    while using a non Identity CTF.

    This case may come up when a developer switches between
    clean and dirty images.
    """
    sim, cov2d, coef_clean, _, h_ctf_fb, h_idx = cov2d_fixture

    coef_cwf = cov2d.get_cwf_coefs(coef_clean, h_ctf_fb, h_idx, noise_var=0)

    # Reconstruct images from output of get_cwf_coefs
    img_est = cov2d.basis.evaluate(coef_cwf)
    # Compare with clean images
    delta = np.mean(np.square((sim.clean_images[:] - img_est).asnumpy()))
    np.testing.assert_array_less(delta, 0.01)


def test_shrinker_inputs(cov2d_fixture):
    """
    Check we raise with specific message for unsupporting shrinker arg.
    """
    cov2d, coef_clean = cov2d_fixture[1], cov2d_fixture[2]

    bad_shrinker_inputs = ["None", "notashrinker", ""]

    for shrinker in bad_shrinker_inputs:
        with raises(AssertionError, match="Unsupported shrink method"):
            _ = cov2d.get_covar(coef_clean, covar_est_opt={"shrinker": shrinker})


def test_shrinkage(cov2d_fixture, shrinker):
    """
    Test all the shrinkers we know about run without crashing,
    """
    cov2d, coef_clean = cov2d_fixture[1], cov2d_fixture[2]

    results = np.load(
        os.path.join(DATA_DIR, "clean70SRibosome_cov2d_covar.npz"),
    )

    covar_coef = cov2d.get_covar(coef_clean, covar_est_opt={"shrinker": shrinker})

    for im, mat in enumerate(results.values()):
        np.testing.assert_allclose(
            mat, covar_coef[im], atol=utest_tolerance(cov2d.dtype)
        )


def test_get_cwf_coefs_ctf_args(cov2d_fixture):
    """
    Test we raise when user supplies incorrect CTF arguments,
    and that the error message matches.
    """
    sim, cov2d, _, coef, h_ctf_fb, _ = cov2d_fixture

    # When half the ctf info (h_ctf_fb) is populated,
    #   set the other ctf param (h_idx) to none.
    h_idx = None
    if h_ctf_fb is None:
        # And when h_ctf_fb is None, we'll populate the other half (h_idx)
        h_idx = sim.filter_indices

    # Both the above situations should be an error.
    with raises(RuntimeError, match=r".*Given ctf_.*"):
        _ = cov2d.get_cwf_coefs(coef, h_ctf_fb, h_idx, noise_var=NOISE_VAR)


def test_get_cwf_coefs(cov2d_fixture, ctf_enabled):
    """
    Tests `get_cwf_coefs` with poulated CTF.
    """
    _, cov2d, coef_clean, coef, h_ctf_fb, h_idx = cov2d_fixture

    # Hard coded file expects sim with ctf.
    if not ctf_enabled:
        pytest.skip(reason="Reference file n/a.")

    results = np.load(os.path.join(DATA_DIR, "clean70SRibosome_cov2d_cwf_coef.npy"))

    coef_cwf = cov2d.get_cwf_coefs(coef, h_ctf_fb, h_idx, noise_var=NOISE_VAR)

    np.testing.assert_allclose(results, coef_cwf, atol=utest_tolerance(cov2d.dtype))


def test_get_cwf_coefs_without_ctf_args(cov2d_fixture, ctf_enabled):
    """
    Tests `get_cwf_coefs` with poulated CTF.
    """

    _, cov2d, _, coef, _, _ = cov2d_fixture

    # Hard coded file expects sim with ctf.
    if not ctf_enabled:
        pytest.skip(reason="Reference file n/a.")

    # Note, I think this file is incorrectly named...
    #   It appears to have come from operations on images with ctf applied.
    results = np.load(
        os.path.join(DATA_DIR, "clean70SRibosome_cov2d_cwf_coef_noCTF.npy")
    )

    coef_cwf = cov2d.get_cwf_coefs(coef, noise_var=NOISE_VAR)

    np.testing.assert_allclose(results, coef_cwf, atol=utest_tolerance(cov2d.dtype))


def test_get_covar_ctf(cov2d_fixture, ctf_enabled):
    # Hard coded file expects sim with ctf.
    if not ctf_enabled:
        pytest.skip(reason="Reference file n/a.")

    sim, cov2d, _, coef, h_ctf_fb, h_idx = cov2d_fixture

    results = np.load(
        os.path.join(DATA_DIR, "clean70SRibosome_cov2d_covarctf.npz"),
    )

    covar_coef_ctf = cov2d.get_covar(coef, h_ctf_fb, h_idx, noise_var=NOISE_VAR)
    for im, mat in enumerate(results.values()):
        # These tolerances were adjusted slightly (1e-8 to 3e-8) to accomodate MATLAB CTF repro changes
        np.testing.assert_allclose(mat, covar_coef_ctf[im], rtol=3e-05, atol=3e-08)


def test_get_covar_ctf_shrink(cov2d_fixture, ctf_enabled):
    sim, cov2d, _, coef, h_ctf_fb, h_idx = cov2d_fixture

    # Hard coded file expects sim with ctf.
    if not ctf_enabled:
        pytest.skip(reason="Reference file n/a.")

    results = np.load(
        os.path.join(DATA_DIR, "clean70SRibosome_cov2d_covarctf_shrink.npz"),
    )

    covar_opt = {
        "shrinker": "frobenius_norm",
        "verbose": 0,
        "max_iter": 250,
        "iter_callback": [],
        "store_iterates": False,
        "rel_tolerance": 1e-12,
        "precision": cov2d.dtype,
    }

    covar_coef_ctf_shrink = cov2d.get_covar(
        coef,
        h_ctf_fb,
        h_idx,
        noise_var=NOISE_VAR,
        covar_est_opt=covar_opt,
    )

    for im, mat in enumerate(results.values()):
        np.testing.assert_allclose(mat, covar_coef_ctf_shrink[im])
