import numpy as np
import pytest

from aspire.image import Image
from aspire.operators import PolarFT
from aspire.utils import gaussian_2d, grid_2d
from aspire.volume import AsymmetricVolume, CnSymmetricVolume

# ==========
# Parameters
# ==========

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


# ==================
# Parameter Fixtures
# ==================


@pytest.fixture(params=DTYPES, ids=lambda x: f"dtype={x}")
def dtype(request):
    return request.param


@pytest.fixture(params=IMG_SIZES, ids=lambda x: f"img_size={x}")
def img_size(request):
    return request.param


@pytest.fixture(params=RADIAL_MODES, ids=lambda x: f"radial_mode={x}")
def radial_mode(request):
    return request.param


# =====================
# Image and PF Fixtures
# =====================


@pytest.fixture
def gaussian(img_size, dtype):
    """Radially symmetric image."""
    gauss = Image(
        gaussian_2d(img_size, sigma=(img_size // 10, img_size // 10), dtype=dtype)
    )
    pf = pf_transform(gauss)

    return pf


@pytest.fixture
def symmetric_image(img_size, dtype):
    """Cyclically (C4) symmetric image."""
    symmetric_vol = CnSymmetricVolume(
        img_size, C=1, order=4, K=25, seed=10, dtype=dtype
    ).generate()
    symmetric_image = symmetric_vol.project(np.eye(3, dtype=dtype))
    pf = pf_transform(symmetric_image)

    return pf


@pytest.fixture
def asymmetric_image(img_size, dtype):
    """Asymetric image."""
    asymmetric_vol = AsymmetricVolume(img_size, C=1, dtype=dtype).generate()
    asymmetric_image = asymmetric_vol.project(np.eye(3, dtype=dtype))
    pf = pf_transform(asymmetric_image)

    return asymmetric_image, pf


@pytest.fixture
def radial_mode_image(img_size, dtype, radial_mode):
    g = grid_2d(img_size, dtype=dtype)
    image = Image(np.sin(radial_mode * np.pi * g["r"]))
    pf = pf_transform(image)

    return pf, radial_mode


# Helper function
def pf_transform(image):
    """Take polar Fourier transform of image."""
    img_size = image.resolution
    nrad = img_size // 2
    ntheta = 360
    pft = PolarFT(img_size, nrad=nrad, ntheta=ntheta, dtype=image.dtype)
    pf = pft.transform(image)[0]

    return pf


# =============
# Testing Suite
# =============


def test_dc_component(asymmetric_image):
    """Test that the DC component equals the mean of the signal."""
    image, pf = asymmetric_image
    signal_mean = np.mean(image)
    dc_components = abs(pf[:, 0])

    assert np.allclose(dc_components, signal_mean)


def test_radially_symmetric_image(gaussian):
    """Test that all polar Fourier rays are equal for a radially symmetric image."""
    pf = gaussian

    assert np.allclose(pf, pf[0])


def test_cyclically_symmetric_image(symmetric_image):
    """Test that a symmetric image produces repeated sets of polar Fourier rays."""
    pf = symmetric_image

    # For C4 symmetry any two sets of rays seperated by 90 degrees should be equal.
    ntheta = pf.shape[0]  # ntheta is the number of rays in 180 degrees.

    assert np.allclose(abs(pf[: ntheta // 2]), abs(pf[ntheta // 2 :]), atol=1e-7)


def test_radial_modes(radial_mode_image):
    pf, mode = radial_mode_image

    # Set DC component to zero.
    pf[:, 0] = 0

    # Check that all rays are close.
    assert abs(abs(pf) - abs(pf[0])).all() < 1e-4

    # Check that correct mode is most prominent.
    # Mode could be off by a pixel depending on resolution and mode.
    # Since all rays are close will just test one.
    mode_window = [mode - 1, mode, mode + 1]
    ray = 3
    assert np.argmax(abs(pf[ray])) in mode_window


def test_complex_image_error():
    """Test that we raise for complex images."""
    img_size = 5
    complex_image = Image(np.ones((img_size, img_size), dtype=np.complex64)) + 2j
    pft = PolarFT(size=img_size, dtype=np.complex64)
    with pytest.raises(TypeError, match=r"The Image `x` must be real valued*"):
        _ = pft.transform(complex_image)


def test_numpy_array_error():
    """Test that we raise when passed numpy array."""
    img_size = 5
    image_np = np.ones((img_size, img_size), dtype=np.float32)
    pft = PolarFT(size=img_size, dtype=np.float32)
    with pytest.raises(TypeError, match=r"passed numpy array*"):
        _ = pft.transform(image_np)


def test_inconsistent_dtypes_error():
    """Test that we raise for complex images."""
    img_size = 5
    image = Image(np.ones((img_size, img_size), dtype=np.float32))
    pft = PolarFT(size=img_size, dtype=np.float64)
    with pytest.raises(TypeError, match=r"Inconsistent dtypes*"):
        _ = pft.transform(image)


def test_theta_error():
    """
    Test that `PolarFT`, when instantiated with odd value for `ntheta`,
    gives appropriate error.
    """

    # Test we raise with expected error.
    with pytest.raises(NotImplementedError, match=r"Only even values for ntheta*"):
        _ = PolarFT(size=42, ntheta=143, dtype=np.float32)


@pytest.mark.parametrize("stack_shape", [(5,), (2, 3)])
def test_half_to_full_transform(stack_shape):
    """
    Test conjugate symmetry and shape of the full polar Fourier transform.
    """
    img_size = 32
    image = Image(
        np.random.rand(*stack_shape, img_size, img_size).astype(np.float32, copy=False)
    )
    pft = PolarFT(size=img_size)
    pf = pft.transform(image)
    full_pf = pft.half_to_full(pf)

    # Check shape.
    assert full_pf.shape == (*stack_shape, pft.ntheta, pft.nrad)

    # Check conjugate symmetry against pf.
    assert np.allclose(np.conj(pf), full_pf[..., pft.ntheta // 2 :, :])

    # Check conjugate symmetry against self.
    for ray in range(pft.ntheta // 2):
        np.testing.assert_allclose(
            full_pf[..., ray, :], np.conj(full_pf[..., ray + pft.ntheta // 2, :])
        )
