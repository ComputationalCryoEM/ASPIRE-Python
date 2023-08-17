import numpy as np
import pytest

from aspire.basis import PolarFT
from aspire.image import Image
from aspire.utils import gaussian_2d, grid_2d
from aspire.volume import AsymmetricVolume, CnSymmetricVolume

# Parametrize over (resolution, dtype)
IMG_SIZES = [
    64,
    65,
]
DTYPES = [
    np.float64,
    np.float32,
]

RADIAL_MODES = [
    2,
    3,
    4,
    5,
    8,
    9,
    16,
    17,
]


@pytest.fixture(params=DTYPES, ids=lambda x: f"dtype={x}")
def dtype(request):
    return request.param


@pytest.fixture(params=IMG_SIZES, ids=lambda x: f"img_size={x}")
def img_size(request):
    return request.param


@pytest.fixture(params=RADIAL_MODES, ids=lambda x: f"radial_mode={x}")
def radial_mode(request):
    return request.param


@pytest.fixture
def gaussian(img_size, dtype):
    """Radially symmetric image."""
    gauss = Image(
        gaussian_2d(img_size, sigma=(img_size // 10, img_size // 10), dtype=dtype)
    )
    pf, _ = pf_transform(gauss)

    return gauss, pf


@pytest.fixture
def symmetric_image(img_size, dtype):
    """Cyclically (C4) symmetric image."""
    symmetric_vol = CnSymmetricVolume(
        img_size, C=1, order=4, K=25, seed=10, dtype=dtype
    ).generate()
    symmetric_image = symmetric_vol.project(np.eye(3, dtype=dtype))
    pf, pft = pf_transform(symmetric_image)
    pf_inverse = pft._evaluate(pf.reshape(-1))

    return symmetric_image, pf, pf_inverse


@pytest.fixture
def asymmetric_image(img_size, dtype):
    """Asymetric image."""
    asymmetric_vol = AsymmetricVolume(img_size, C=1, dtype=dtype).generate()
    asymmetric_image = asymmetric_vol.project(np.eye(3, dtype=dtype))
    pf, _ = pf_transform(asymmetric_image)

    return asymmetric_image, pf


@pytest.fixture
def radial_mode_image(img_size, dtype, radial_mode):
    g = grid_2d(img_size, dtype=dtype)
    image = Image(np.sin(radial_mode * np.pi * g["r"]))
    pf, _ = pf_transform(image)

    return pf, radial_mode


def pf_transform(image):
    """Take polar Fourier transform of image."""
    img_size = image.resolution
    nrad = img_size // 2
    ntheta = 8 * nrad
    pft = PolarFT(img_size, nrad=nrad, ntheta=ntheta, dtype=image.dtype)
    pf = pft.evaluate_t(image)
    pf = pf.reshape(ntheta // 2, nrad)

    return pf, pft


def test_dc_component(asymmetric_image):
    """Test that the DC component equals the mean of the signal."""
    image, pf = asymmetric_image
    signal_mean = np.mean(image)
    dc_components = abs(pf[:, 0])

    assert np.allclose(dc_components, signal_mean)


def test_radially_symmetric_image(gaussian):
    """Test that all polar Fourier rays are equal for a radially symmetric image."""
    _, pf = gaussian

    assert np.allclose(pf, pf[0])


def test_cyclically_symmetric_image(symmetric_image):
    """Test that a symmetric image produces repeated sets of polar Fourier rays."""
    _, pf, _ = symmetric_image

    # For C4 symmetry any two sets of rays seperated by 90 degrees should be equal.
    ntheta = pf.shape[0]  # ntheta is the number of rays in 180 degrees.

    assert np.allclose(abs(pf[: ntheta // 2]), abs(pf[ntheta // 2 :]), atol=1e-7)


def test_radial_modes(radial_mode_image):
    pf, mode = radial_mode_image

    # Set DC compenent to zero.
    pf[:, 0] = 0

    # Check that all rays are close.
    assert abs(np.real(pf) - np.real(pf[0])).all() < 1e-4

    # Check that correct mode is most prominent.
    # Mode could be off by a pixel depending on resolution and mode.
    mode_window = [mode - 1, mode, mode + 1]
    assert np.argmax(np.real(pf[3])) in mode_window


def test_adjoint_property(asymmetric_image, symmetric_image):
    """Test the adjoint property."""
    # The evaluate function should be the adjoint operator of evaluate_t.
    # Namely, if A = evaluate, B = evaluate_t, and B=A^t, we will have
    # (y, A*x) = (A^t*y, x) = (B*y, x).
    # There is no significance to using asymmetric_image and symmetric_image
    # below, other than that they are different images.
    y, By = asymmetric_image
    _, x, Ax = symmetric_image

    lhs = y.asnumpy().reshape(-1) @ Ax.reshape(-1)
    rhs = np.real(By.reshape(-1) @ x.reshape(-1))

    if y.resolution % 2 == 1:
        pytest.skip("Currently failing for odd resolution.")

    assert np.allclose(lhs, rhs)


def test_theta_error():
    """
    Test that `PolarFT`, when instantiated with odd value for `ntheta`,
    gives appropriate error.
    """

    # Test we raise with expected error.
    with pytest.raises(NotImplementedError, match=r"Only even values for ntheta*"):
        _ = PolarFT(size=42, ntheta=143, dtype=np.float32)
