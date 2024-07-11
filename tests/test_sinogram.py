import numpy as np
import pytest
from skimage import data
from skimage.transform import radon

from aspire.image import Image
from aspire.utils import grid_2d

# Relative tolerance comparing line projections to scikit
# The same tolerance will be used in all scikit comparisons
SK_TOL = 0.002

IMG_SIZES = [
    511,
    512,
]

DTYPES = [
    np.float32,
    np.float64,
]

ANGLES = [1, 50, 90, 117, 180, 360]

@pytest.fixture(params=DTYPES, ids=lambda x: f"dtype={x}", scope="module")
def dtype(request):
    """
    Dtypes for image.
    """
    return request.param


@pytest.fixture(params=IMG_SIZES, ids=lambda x: f"px={x}", scope="module")
def img_size(request):
    """
    Image size.
    """
    return request.param

@pytest.fixture(params=ANGLES, ids=lambda x: f"angles={x}", scope="module")
def num_ang(request):
    """
    Angles.
    """
    return request.param


@pytest.fixture
def masked_image(dtype, img_size):
    """
    Creates a masked image fixture using camera data from Skikit-Image.
    """
    g = grid_2d(img_size, normalized=True, shifted=True)
    mask = g["r"] < 1

    image = data.camera().astype(dtype)
    image = image[:img_size, :img_size]
    return Image(image * mask)


# Image.project and compare results to skimage.radon
def test_image_project(masked_image, num_ang):
    """
    Test Image.project on a single stack of images. Compares project method with skimage.
    """
    ny = masked_image.resolution
    angles = np.linspace(0, 360, num_ang, endpoint=False)
    rads = angles / 180 * np.pi
    s = masked_image.project(rads)
    assert s.shape == (1, len(angles), ny)

    # sci-kit image `radon` reference
    #
    # Note, Image.project's angles are wrt projection line (ie
    # grid), while sk's radon are wrt the image. To correspond the
    # rotations are inverted.  This was the convention prefered by
    # the original author of this method.
    #
    # Note, transpose sk output to match (angles, points)
    reference_sinogram = radon(masked_image._data[0], theta=angles[::-1]).T
    assert reference_sinogram.shape == (len(angles), ny), "Incorrect Shape"

    # compare project method on ski-image reference
    nrms = np.sqrt(np.mean((s[0] - reference_sinogram) ** 2, axis=1)) / np.linalg.norm(
        reference_sinogram, axis=1
    )

    np.testing.assert_array_less(nrms, SK_TOL, "Error in image projections.")


def test_multidim():
    """
    Test Image.project on stacks of images. Extension of test_image_project but for multi-dimensional stacks.
    """

    L = 512  # pixels
    n = 3

    # Generate a mask
    g = grid_2d(L, normalized=True, shifted=True)
    mask = g["r"] < 1

    # Generate images
    imgs = Image(np.random.random((n, L, L))) * mask

    # Generate line project angles
    angles = np.linspace(0, 180, L, endpoint=False)
    rads = angles / 180.0 * np.pi
    s = imgs.project(rads)

    # Compare
    reference_sinograms = np.empty((n, L, L))
    for i, img in enumerate(imgs):
        # Compute the singleton case, and compare with the stack
        single_sinogram = img.project(rads)
        # These should be allclose up to determinism in the FFT and NUFFT.
        np.testing.assert_allclose(s[i : i + 1], single_sinogram)

        # Next individually compute sk's radon transform for each image.
        reference_sinograms[i] = radon(img._data[0], theta=angles[::-1]).T

    # Compare all lines in each sinogram with sk-image
    for i in range(n):
        nrms = np.sqrt(
            np.mean((s[i] - reference_sinograms[i]) ** 2, axis=1)
        ) / np.linalg.norm(reference_sinograms[i], axis=1)
        np.testing.assert_array_less(nrms, SK_TOL, err_msg=f"Error in image {i}.")
