import logging

import numpy as np
import PIL.Image as PILImage
import pytest

from aspire.basis import FBBasis2D, FFBBasis2D, FLEBasis2D, FPSWFBasis2D, PSWFBasis2D
from aspire.image import Image
from aspire.utils import gaussian_2d

logger = logging.getLogger(__name__)


# Parameters

DTYPES = [
    np.float32,
    pytest.param(np.float64, marks=pytest.mark.expensive),
]

BASES = [
    FFBBasis2D,
    FBBasis2D,
    FLEBasis2D,
    PSWFBasis2D,
    FPSWFBasis2D,
]

IMG_SIZES = [
    31,
    pytest.param(32, marks=pytest.mark.expensive),
]

# Fixtures


@pytest.fixture(params=DTYPES, ids=lambda x: f"dtype={x}", scope="module")
def dtype(request):
    return request.param


@pytest.fixture(params=IMG_SIZES, ids=lambda x: f"img_size={x}", scope="module")
def img_size(request):
    return request.param


@pytest.fixture(params=BASES, ids=lambda x: f"basis={x}", scope="module")
def basis(request, img_size, dtype):
    cls = request.param
    # Setup a Basis
    basis = cls(img_size, dtype=dtype)
    return basis


# Basis Rotations


def test_basis_rotation_2d(basis):
    """
    Test steerable basis rotation performs similar operation to PIL real space image rotation.

    Checks both orientation and rough values.
    """
    # Set a rotation amount
    rot_radians = np.pi / 6

    # Create an Image containing a smooth blob.
    L = basis.nres
    img = gaussian_2d(L, mu=(L // 4, 0), dtype=basis.dtype)
    img = Image(img / np.linalg.norm(img), pixel_size=1.234)  # Normalize

    # Rotate with an ASPIRE steerable basis, returning to real space.
    rot_img = basis.expand(img).rotate(rot_radians).evaluate()

    # Rotate image with PIL, returning to Numpy array.
    pil_rot_img = np.asarray(
        PILImage.fromarray(img.asnumpy()[0]).rotate(
            rot_radians * 180 / np.pi, resample=PILImage.BICUBIC
        )
    )

    # Rough compare arrays.
    np.testing.assert_allclose(rot_img.asnumpy()[0], pil_rot_img, atol=0.15)

    # Test pixel_size passthrough
    np.testing.assert_array_equal(img.pixel_size, rot_img.pixel_size)


def test_basis_reflection_2d(basis):
    """
    Test steerable basis reflection performs similar operation to Numpy flips.

    Checks both orientation and rough values.
    """

    # Create an Image containing a smooth blob.
    L = basis.nres
    img = gaussian_2d(L, mu=(L // 4, L // 5), dtype=basis.dtype)
    img = Image(img / np.linalg.norm(img))  # Normalize

    # Reflect with an ASPIRE steerable basis, returning to real space.
    refl_img = basis.expand(img).rotate(0, refl=True).evaluate()

    # Reflect image with Numpy.
    # Note for odd images we can accurately use Numpy,
    #   but evens have the expected offset issue
    #   when compared to a purely row/col based flip.
    flip = np.flipud
    if isinstance(basis, PSWFBasis2D):
        # TODO, reconcile PSWF reflection axis
        flip = np.fliplr

    refl_img_np = flip(img.asnumpy()[0])

    # Rough compare arrays.
    atol = 0.01
    if L % 2 == 0:
        # Even images test is crude,
        # but is enough ensure flipping without complicating test.
        atol = 0.5

    np.testing.assert_allclose(refl_img.asnumpy()[0], refl_img_np, atol=atol)
