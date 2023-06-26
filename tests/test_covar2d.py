import os
import os.path

import numpy as np
import pytest
from pytest import raises

from aspire.basis import FFBBasis2D
from aspire.covariance import RotCov2D
from aspire.noise import WhiteNoiseAdder
from aspire.operators import RadialCTFFilter
from aspire.source.simulation import Simulation
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

# These variables support parameterized arg checking in `test_shrinkage`
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
        np.load(os.path.join(DATA_DIR, "clean70SRibosome_vol.npy")).astype(dtype)
    )
    # 1e3 is hardcoded to match legacy test files.
    return v.downsample(img_size) * 1.0e3


@pytest.fixture(params=BASIS, ids=lambda x: f"basis={x}")
def basis(request, img_size, dtype):
    cls = request.param
    # Setup a Basis
    basis = cls(img_size, dtype=dtype)
    return basis


@pytest.fixture
def cov2d_fixture(volume, basis, ctf_enabled):
    """
    Cov2D Test without CTFFilters populated.
    """
    n = 32

    # Default CTF params
    unique_filters = None
    h_idx = None
    h_ctf_fb = None
    # Popluate CTF
    if ctf_enabled:
        unique_filters = [
            RadialCTFFilter(
                5.0 * 65 / volume.resolution, 200, defocus=d, Cs=2.0, alpha=0.1
            )
            for d in np.linspace(1.5e4, 2.5e4, 7)
        ]

        # Copied from simulation defaults to match legacy test files.
        h_idx = randi(len(unique_filters), n, seed=0) - 1

        h_ctf_fb = [basis.filter_to_basis_mat(f) for f in unique_filters]

    noise_adder = WhiteNoiseAdder(var=NOISE_VAR)

    sim = Simulation(
        n=n,
        vols=volume,
        unique_filters=unique_filters,
        filter_indices=h_idx,
        offsets=0.0,
        amplitudes=1.0,
        dtype=volume.dtype,
        noise_adder=noise_adder,
    )
    sim.cache()

    # XXX, remove, keeping tmp for reference
    # imgs_clean = sim.projections[:]
    # imgs_ctf_clean = sim.clean_images[:]
    # imgs_ctf_noise = sim.images[:n]

    cov2d = RotCov2D(basis)
    coeff_clean = basis.evaluate_t(sim.projections[:])
    coeff = basis.evaluate_t(sim.images[:])

    return sim, cov2d, coeff_clean, coeff, h_ctf_fb, h_idx


def test_get_mean(cov2d_fixture):
    results = np.load(os.path.join(DATA_DIR, "clean70SRibosome_cov2d_mean.npy"))
    cov2d, coeff_clean = cov2d_fixture[1], cov2d_fixture[2]

    mean_coeff = cov2d._get_mean(coeff_clean.asnumpy())
    assert np.allclose(results, mean_coeff)


def test_get_covar(cov2d_fixture):
    results = np.load(
        os.path.join(DATA_DIR, "clean70SRibosome_cov2d_covar.npy"),
        allow_pickle=True,
    )

    cov2d, coeff_clean = cov2d_fixture[1], cov2d_fixture[2]
    covar_coeff = cov2d._get_covar(coeff_clean.asnumpy())

    for im, mat in enumerate(results.tolist()):
        assert np.allclose(mat, covar_coeff[im])


def test_get_mean_ctf(cov2d_fixture, ctf_enabled):
    """
    Compare `get_mean` (no CTF args) with `_get_mean` (no CTF model).
    """
    sim, cov2d, coeff_clean, coeff, h_ctf_fb, h_idx = cov2d_fixture

    mean_coeff_ctf = cov2d.get_mean(coeff, h_ctf_fb, h_idx)

    if ctf_enabled:
        result = np.load(os.path.join(DATA_DIR, "clean70SRibosome_cov2d_meanctf.npy"))
    else:
        result = cov2d._get_mean(coeff_clean.asnumpy())

    assert np.allclose(mean_coeff_ctf, result, atol=0.002)


def test_get_cwf_coeffs_clean(cov2d_fixture):
    results = np.load(
        os.path.join(DATA_DIR, "clean70SRibosome_cov2d_cwf_coeff_clean.npy")
    )

    cov2d, coeff_clean = cov2d_fixture[1], cov2d_fixture[2]

    coeff_cwf_clean = cov2d.get_cwf_coeffs(coeff_clean, noise_var=0)
    assert np.allclose(results, coeff_cwf_clean, atol=utest_tolerance(cov2d.dtype))


def test_get_cwf_coeffs_clean_ctf(cov2d_fixture):
    """
    Test case of clean images (coeff_clean and noise_var=0)
    while using a non Identity CTF.

    This case may come up when a developer switches between
    clean and dirty images.
    """
    sim, cov2d, coeff_clean, _, h_ctf_fb, h_idx = cov2d_fixture

    coeff_cwf = cov2d.get_cwf_coeffs(coeff_clean, h_ctf_fb, h_idx, noise_var=0)

    # Reconstruct images from output of get_cwf_coeffs
    img_est = cov2d.basis.evaluate(coeff_cwf)
    # Compare with clean images
    delta = np.mean(np.square((sim.projections[:] - img_est).asnumpy()))
    assert delta < 0.02


def test_shrinker_inputs(cov2d_fixture):
    """
    Check we raise with specific message for unsupporting shrinker arg.
    """
    cov2d, coeff_clean = cov2d_fixture[1], cov2d_fixture[2]

    bad_shrinker_inputs = ["None", "notashrinker", ""]

    for shrinker in bad_shrinker_inputs:
        with raises(AssertionError, match="Unsupported shrink method"):
            _ = cov2d.get_covar(coeff_clean, covar_est_opt={"shrinker": shrinker})


def test_shrinkage(cov2d_fixture, shrinker):
    """
    Test all the shrinkers we know about run without crashing,
    """
    cov2d, coeff_clean = cov2d_fixture[1], cov2d_fixture[2]

    results = np.load(
        os.path.join(DATA_DIR, "clean70SRibosome_cov2d_covar.npy"),
        allow_pickle=True,
    )

    covar_coeff = cov2d.get_covar(coeff_clean, covar_est_opt={"shrinker": shrinker})

    for im, mat in enumerate(results.tolist()):
        assert np.allclose(mat, covar_coeff[im], atol=utest_tolerance(cov2d.dtype))


def test_get_cwf_coeffs_ctf_args(cov2d_fixture):
    """
    Test we raise when user supplies incorrect CTF arguments,
    and that the error message matches.
    """
    sim, cov2d, _, coeff, h_ctf_fb, _ = cov2d_fixture

    # When half the ctf info (h_ctf_fb) is populated,
    #   set the other ctf param (h_idx) to none.
    h_idx = None
    if h_ctf_fb is None:
        # And when h_ctf_fb is None, we'll populate the other half (h_idx)
        h_idx = sim.filter_indices

    # Both the above situations should be an error.
    with raises(RuntimeError, match=r".*Given ctf_.*"):
        _ = cov2d.get_cwf_coeffs(coeff, h_ctf_fb, h_idx, noise_var=NOISE_VAR)


def test_get_cwf_coeffs(cov2d_fixture, ctf_enabled):
    """
    Tests `get_cwf_coeffs` with poulated CTF.
    """
    _, cov2d, coeff_clean, coeff, h_ctf_fb, h_idx = cov2d_fixture

    # Hard coded file expects sim with ctf.
    if not ctf_enabled:
        pytest.skip(reason="Reference file n/a.")

    results = np.load(os.path.join(DATA_DIR, "clean70SRibosome_cov2d_cwf_coeff.npy"))

    coeff_cwf = cov2d.get_cwf_coeffs(coeff, h_ctf_fb, h_idx, noise_var=NOISE_VAR)

    assert np.allclose(results, coeff_cwf, atol=utest_tolerance(cov2d.dtype))


def test_get_cwf_coeffs_without_ctf_args(cov2d_fixture, ctf_enabled):
    """
    Tests `get_cwf_coeffs` with poulated CTF.
    """

    _, cov2d, _, coeff, _, _ = cov2d_fixture

    # Hard coded file expects sim with ctf.
    if not ctf_enabled:
        pytest.skip(reason="Reference file n/a.")

    # Note, I think this file is incorrectly named...
    #   It appears to have come from operations on images with ctf applied.
    results = np.load(
        os.path.join(DATA_DIR, "clean70SRibosome_cov2d_cwf_coeff_noCTF.npy")
    )

    coeff_cwf = cov2d.get_cwf_coeffs(coeff, noise_var=NOISE_VAR)

    assert np.allclose(results, coeff_cwf, atol=utest_tolerance(cov2d.dtype))


def test_get_covar_ctf(cov2d_fixture, ctf_enabled):
    # Hard coded file expects sim with ctf.
    if not ctf_enabled:
        pytest.skip(reason="Reference file n/a.")

    sim, cov2d, _, coeff, h_ctf_fb, h_idx = cov2d_fixture

    results = np.load(
        os.path.join(DATA_DIR, "clean70SRibosome_cov2d_covarctf.npy"),
        allow_pickle=True,
    )

    covar_coeff_ctf = cov2d.get_covar(coeff, h_ctf_fb, h_idx, noise_var=NOISE_VAR)
    for im, mat in enumerate(results.tolist()):
        assert np.allclose(mat, covar_coeff_ctf[im])


def test_get_covar_ctf_shrink(cov2d_fixture, ctf_enabled):
    sim, cov2d, _, coeff, h_ctf_fb, h_idx = cov2d_fixture

    # Hard coded file expects sim with ctf.
    if not ctf_enabled:
        pytest.skip(reason="Reference file n/a.")

    results = np.load(
        os.path.join(DATA_DIR, "clean70SRibosome_cov2d_covarctf_shrink.npy"),
        allow_pickle=True,
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

    covar_coeff_ctf_shrink = cov2d.get_covar(
        coeff,
        h_ctf_fb,
        h_idx,
        noise_var=NOISE_VAR,
        covar_est_opt=covar_opt,
    )

    for im, mat in enumerate(results.tolist()):
        assert np.allclose(mat, covar_coeff_ctf_shrink[im])
