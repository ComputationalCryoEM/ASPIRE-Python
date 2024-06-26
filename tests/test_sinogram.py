import numpy as np
import pytest
from skimage import data
from skimage.transform import radon

from aspire.image import Image
from aspire.source import Simulation
from aspire.utils import grid_2d

# parameter img_sizes: 511, 512
IMG_SIZES = [
    511,
    512,
]

# parameter dtype: float32, float64
DTYPES = [
    np.float32,
    np.float64,
]


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


@pytest.fixture
def masked_image(dtype, img_size):
    """
    Construct a masked image fixture that takes paramters
    """
    g = grid_2d(img_size, normalized=True, shifted=True)
    mask = g["r"] < 1

    image = data.camera().astype(dtype)
    image = image[:img_size, :img_size]
    return Image(image * mask)


# Image.project and compare results to skimage.radon
def test_image_project(masked_image):
    """
    TestImage.project on a single stack of images. Compares project method with skimage.
    """
    ny = masked_image.resolution
    angles = np.linspace(0, 360, ny, endpoint=False)
    rads = angles / 180 * np.pi
    s = masked_image.project(rads)

    # add reference skimage radon here
    n = masked_image._data[0]
    reference_sinogram = radon(n, theta=angles[::-1])

    # compare s with reference
    nrms = np.sqrt(np.mean((s[0] - reference_sinogram) ** 2, axis=0)) / np.linalg.norm(
        reference_sinogram, axis=0
    )
    tol = 0.002

    # odd image tolerance (stink)
    if masked_image.resolution % 2 == 1:
        tol = 0.02
    np.testing.assert_array_less(nrms, tol, "Error in test image")


def test_multidim():
    """
    Test Image.project on stacks of images.
    """

    L = 512  # pixels
    n = 3

    # Generate a mask
    g = grid_2d(L, normalized=True, shifted=True)
    mask = g["r"] < 1

    # Generate a simulation
    src = Simulation(n=n, L=L, C=1, dtype=np.float64)
    imgs = src.images[:] * mask

    # Generate line project angles
    angles = np.linspace(0, 180, L, endpoint=False)
    rads = angles / 180.0 * np.pi
    s = imgs.project(rads)

    # # Compare with sk
    reference_sinograms = np.empty((n, L, L))
    for i, img in enumerate(imgs._data):
        reference_sinograms[i] = radon(img, theta=angles[::-1])

    # decrease tolerance as L goes up
    for i in range(n):
        nrms = np.sqrt(
            np.mean((s[i] - reference_sinograms[i]) ** 2, axis=0)
        ) / np.linalg.norm(reference_sinograms[i], axis=0)
        np.testing.assert_array_less(nrms, 0.05, err_msg=f"Error in image {i}")
