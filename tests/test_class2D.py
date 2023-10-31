import logging
import os

import numpy as np
import pytest
from sklearn import datasets

from aspire.basis import (
    Coef,
    FBBasis2D,
    FFBBasis2D,
    FLEBasis2D,
    FPSWFBasis2D,
    FSPCABasis,
    PSWFBasis2D,
)
from aspire.classification import RIRClass2D
from aspire.classification.legacy_implementations import bispec_2drot_large, pca_y
from aspire.noise import WhiteNoiseAdder
from aspire.source import Simulation
from aspire.utils import utest_tolerance
from aspire.volume import Volume

logger = logging.getLogger(__name__)


DATA_DIR = os.path.join(os.path.dirname(__file__), "saved_test_data")

# This seed is to stabilize the extremely small unit test (img size 16 etc).
SEED = 42


IMG_SIZES = [16]
DTYPES = [np.float32]
# Basis used in FSPCA for class averaging.
BASIS = [
    FFBBasis2D,
    pytest.param(FBBasis2D, marks=pytest.mark.expensive),
    pytest.param(FLEBasis2D, marks=pytest.mark.expensive),
    pytest.param(PSWFBasis2D, marks=pytest.mark.expensive),
    pytest.param(FPSWFBasis2D, marks=pytest.mark.expensive),
]


@pytest.fixture(params=DTYPES, ids=lambda x: f"dtype={x}", scope="module")
def dtype(request):
    return request.param


@pytest.fixture(params=IMG_SIZES, ids=lambda x: f"img_size={x}", scope="module")
def img_size(request):
    return request.param


@pytest.fixture(scope="module")
def volume(dtype, img_size):
    # Get a volume
    v = Volume(
        np.load(os.path.join(DATA_DIR, "clean70SRibosome_vol.npy")).astype(dtype)
    )

    return v.downsample(img_size)


@pytest.fixture(scope="module")
def sim_fixture(volume, img_size, dtype):
    """
    Provides a clean simulation parameterized by `img_size` and `dtype`.
    """

    # Create a src from the volume
    src = Simulation(L=img_size, n=321, vols=volume, dtype=dtype, seed=SEED)
    src = src.cache()  # Precompute image stack

    # Calculate some projection images
    imgs = src.images[:]

    # Configure an FSPCA basis
    fspca_basis = FSPCABasis(src, noise_var=0)

    return imgs, src, fspca_basis


@pytest.fixture(params=BASIS, ids=lambda x: f"basis={x}", scope="module")
def basis(request, img_size, dtype):
    cls = request.param
    # Setup a Basis
    basis = cls(img_size, dtype=dtype)
    return basis


def test_expand_eval(sim_fixture):
    imgs, _, fspca_basis = sim_fixture
    coef = fspca_basis.expand_from_image_basis(imgs)
    recon = fspca_basis.evaluate_to_image_basis(coef)

    # Check recon is close to imgs
    rmse = np.sqrt(np.mean(np.square(imgs.asnumpy() - recon.asnumpy())))
    logger.info(f"FSPCA Expand Eval Image Round True RMSE: {rmse}")
    assert rmse < utest_tolerance(fspca_basis.dtype)


def test_complex_conversions_errors(sim_fixture):
    """
    Test we raise when passed incorrect dtypes.

    Also checks we can handle 0d vector in `to_real`.

    Most other cases covered by classification unit tests.
    """
    imgs, _, fspca_basis = sim_fixture

    with pytest.raises(TypeError):
        _ = fspca_basis.to_complex(
            Coef(
                fspca_basis,
                np.arange(fspca_basis.count),
                dtype=np.complex64,
            )
        )

    with pytest.raises(TypeError):
        _ = fspca_basis.to_real(
            Coef(fspca_basis, np.arange(fspca_basis.count), dtype=np.float32)
        )


def test_rotate(sim_fixture):
    """
    Trivial test of rotation in FSPCA Basis.

    Also covers to_real and to_complex conversions in FSPCA Basis.
    """
    imgs, _, fspca_basis = sim_fixture

    coef = fspca_basis.expand_from_image_basis(imgs)
    # rotate by pi
    rot_coef = fspca_basis.rotate(coef, radians=np.pi)
    rot_imgs = fspca_basis.evaluate_to_image_basis(rot_coef)

    for i, img in enumerate(imgs):
        rmse = np.sqrt(np.mean(np.square(np.flip(img) - rot_imgs[i])))
        assert rmse < 10 * utest_tolerance(fspca_basis.dtype)


def test_basis_too_small(sim_fixture, basis):
    """
    When number of components is more than basis functions raise with descriptive error.
    """
    src = sim_fixture[1]

    with pytest.raises(ValueError, match=r".*Reduce components.*"):
        # Configure an FSPCA basis
        _ = FSPCABasis(src, basis=basis, components=basis.count * 2, noise_var=0)


@pytest.fixture(scope="module")
def sim_fixture2(volume, basis, img_size, dtype):
    """
    Provides clean/noisy pair of smaller parameterized simulations,
    along with corresponding clean/noisy basis and an additional
    compressed basis.

    These are slightly smaller than `sim_fixture` and support covering
    additional code and corner cases.
    """

    n_img = 150

    # Clean
    clean_src = Simulation(L=img_size, n=n_img, vols=volume, dtype=dtype, seed=SEED)
    clean_src = clean_src.cache()

    # With Noise
    noise_var = 0.01 * np.var(np.sum(volume[0], axis=0))
    noise_adder = WhiteNoiseAdder(var=noise_var)
    noisy_src = Simulation(
        L=img_size,
        n=n_img,
        vols=volume,
        dtype=dtype,
        noise_adder=noise_adder,
        seed=SEED,
    )
    noisy_src = noisy_src.cache()

    # Create Basis, use precomputed Basis
    clean_fspca_basis = FSPCABasis(
        clean_src, basis, noise_var=0
    )  # Note noise_var assigned zero, skips eigval filtering.

    clean_fspca_basis_compressed = FSPCABasis(
        clean_src, basis, components=101, noise_var=0
    )  # Note noise_var assigned zero, skips eigval filtering.

    # Ceate another fspca_basis, use autogeneration Basis
    noisy_fspca_basis = FSPCABasis(noisy_src)

    return (
        clean_src,
        noisy_src,
        clean_fspca_basis,
        clean_fspca_basis_compressed,
        noisy_fspca_basis,
    )


def test_source_too_small(sim_fixture2):
    """
    When number of images in source is less than requested bispectrum components,
    raise with descriptive error.
    """
    clean_src = sim_fixture2[0]

    with pytest.raises(
        RuntimeError, match=r".*Increase number of images or reduce components.*"
    ):
        _ = RIRClass2D(
            clean_src,
            fspca_components=clean_src.n * 4,
            bispectrum_components=clean_src.n * 2,
        )


def test_incorrect_components(sim_fixture2):
    """
    Check we raise with inconsistent configuration of FSPCA components.
    """
    clean_src, clean_fspca_basis = sim_fixture2[0], sim_fixture2[2]

    with pytest.raises(
        RuntimeError, match=r"`pca_basis` components.*provided by user."
    ):
        _ = RIRClass2D(
            clean_src,
            clean_fspca_basis,  # 400 components
            fspca_components=100,
            large_pca_implementation="legacy",
            nn_implementation="legacy",
            bispectrum_implementation="legacy",
        )

    # Explicitly providing the same number should be okay.
    _ = RIRClass2D(
        clean_src,
        clean_fspca_basis,  # 400 components
        fspca_components=clean_fspca_basis.components,
        bispectrum_components=100,
        large_pca_implementation="legacy",
        nn_implementation="legacy",
        bispectrum_implementation="legacy",
        seed=SEED,
    )


def test_RIR_legacy(basis, sim_fixture2):
    """
    Currently just tests for runtime errors.
    """
    clean_src = sim_fixture2[0]

    clean_fspca_basis = FSPCABasis(
        clean_src, basis, noise_var=0, components=100
    )  # Note noise_var assigned zero, skips eigval filtering.

    rir = RIRClass2D(
        clean_src,
        clean_fspca_basis,
        bispectrum_components=42,
        large_pca_implementation="legacy",
        nn_implementation="legacy",
        bispectrum_implementation="legacy",
        seed=SEED,
    )

    _ = rir.classify()


def test_RIR_devel_disp(sim_fixture2):
    """
    Currently just tests for runtime errors.
    """
    clean_src, fspca_basis = sim_fixture2[0], sim_fixture2[3]

    # Use the basis class setup, only requires a Source.
    rir = RIRClass2D(
        clean_src,
        fspca_components=fspca_basis.components,
        bispectrum_components=fspca_basis.components - 1,
        large_pca_implementation="legacy",
        nn_implementation="legacy",
        bispectrum_implementation="devel",
    )

    _ = rir.classify()


def test_RIR_sk(sim_fixture2):
    """
    Excercises the eigenvalue based filtering,
    along with other swappable components.

    Currently just tests for runtime errors.
    """
    noisy_src, noisy_fspca_basis = sim_fixture2[1], sim_fixture2[4]

    rir = RIRClass2D(
        noisy_src,
        noisy_fspca_basis,
        bispectrum_components=100,
        sample_n=42,
        large_pca_implementation="sklearn",
        nn_implementation="sklearn",
        bispectrum_implementation="devel",
        seed=SEED,
    )

    _ = rir.classify()


def test_eigein_images(sim_fixture2):
    """
    Test we can return eigenimages.
    """
    clean_fspca_basis, clean_fspca_basis_compressed = sim_fixture2[2], sim_fixture2[3]

    # Get the eigenimages from an FSPCA basis for testing
    eigimg_uncompressed = clean_fspca_basis.eigen_images()

    # Get the eigenimages from a compressed FSPCA basis for testing
    eigimg_compressed = clean_fspca_basis_compressed.eigen_images()

    # Check they are close.
    # Note it is expected the compression reorders the eigvecs,
    #  and thus the eigimages.
    # We sum over all the eigimages to yield an "average" for comparison
    assert np.allclose(
        np.sum(eigimg_uncompressed.asnumpy(), axis=0),
        np.sum(eigimg_compressed.asnumpy(), axis=0),
        atol=utest_tolerance(clean_fspca_basis.dtype),
    )


def test_component_size(sim_fixture2):
    """
    Tests we raise when number of components are too small.

    Also tests dtype mismatch behavior.
    """
    clean_src, compressed_fspca_basis = sim_fixture2[0], sim_fixture2[3]

    with pytest.raises(RuntimeError, match=r".*Reduce bispectrum_components.*"):
        _ = RIRClass2D(
            clean_src,
            compressed_fspca_basis,
            bispectrum_components=clean_src.n + 1,
        )


def test_implementations(basis, sim_fixture2):
    """
    Test optional implementations handle bad inputs with a descriptive error.
    """
    clean_src, clean_fspca_basis = sim_fixture2[0], sim_fixture2[2]

    # Nearest Neighbhor component
    with pytest.raises(ValueError, match=r"Provided nn_implementation.*"):
        _ = RIRClass2D(
            clean_src,
            clean_fspca_basis,
            bispectrum_components=int(0.75 * clean_fspca_basis.basis.count),
            nn_implementation="badinput",
        )

    # Large PCA component
    with pytest.raises(ValueError, match=r"Provided large_pca_implementation.*"):
        _ = RIRClass2D(
            clean_src,
            clean_fspca_basis,
            large_pca_implementation="badinput",
        )

    # Bispectrum component
    with pytest.raises(ValueError, match=r"Provided bispectrum_implementation.*"):
        _ = RIRClass2D(
            clean_src,
            clean_fspca_basis,
            bispectrum_implementation="badinput",
        )

    # Legacy Bispectrum implies legacy bispectrum (they're integrated).
    with pytest.raises(
        ValueError, match=r'"legacy" bispectrum_implementation implies.*'
    ):
        _ = RIRClass2D(
            clean_src,
            clean_fspca_basis,
            bispectrum_implementation="legacy",
            large_pca_implementation="sklearn",
        )

    # Currently we only FSPCA Basis in RIRClass2D
    with pytest.raises(
        RuntimeError,
        match="RIRClass2D has currently only been developed for pca_basis as a FSPCABasis.",
    ):
        _ = RIRClass2D(clean_src, basis)


# Cover branches of Legacy code not taken by the classification unit tests.


def test_pca_y():
    """
    We want to check that real inputs and differing input matrix shapes work.

    Most of pca_y is covered by the classificiation unit tests.
    """

    # The iris dataset is a small 150 sample by 5 feature dataset in float64
    iris = datasets.load_iris()

    # Extract the data matrix, run once as is (150, 5),
    # and once tranposed  so shape[0] < shape[1] (5, 150)
    for x in (iris.data, iris.data.T):
        # Run pca_y and check reconstruction holds
        lsvec, svals, rsvec = pca_y(x, 5)

        # svd ~~> A = U S V = (U S) V
        recon = np.dot(lsvec * svals, rsvec)

        assert np.allclose(x, recon)


def test_bispect_overflow():
    """
    A zero value coef will cause a div0 error in log call.
    Check it is raised.
    """

    with pytest.raises(ValueError, match="coef_norm should not be -inf"):
        # This should emit a warning before raising
        with pytest.warns(RuntimeWarning):
            bispec_2drot_large(
                coef=np.arange(10),
                freqs=np.arange(1, 11),
                eigval=np.arange(10),
                alpha=1 / 3,
                sample_n=4000,
            )
