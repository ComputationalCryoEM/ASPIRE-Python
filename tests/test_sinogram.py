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

    # add more logic to check the sizes and readjust accordingly
    image = data.camera().astype(dtype)
    image = image[:img_size, :img_size]
    return Image(image * mask)


# Image.project and compare results to skimage.radon
def test_image_project(masked_image):
    ny = masked_image.resolution
    angles = np.linspace(0, 360, ny, endpoint=False)
    rads = angles / 180 * np.pi
    s = masked_image.project(rads)

    # add reference skimage radon here
    n = masked_image._data[0]
    reference_sinogram = radon(n, theta=angles[::-1])

    # compare s with reference
    np.testing.assert_allclose(s, reference_sinogram, rtol=11, atol=1e-8)

    # create fixture called masked_image(img_size) -> return: masked image of size (grid generation goes in fixture)


def test_multidim():
    """
    Test Image.project on stacks of images.
    """

    L = 32  # pixels
    n = 3

    # Generate a mask
    g = grid_2d(L, normalized=True, shifted=True)
    mask = g["r"] < 1

    # Generate a simulation
    src = Simulation(n=n, L=L, C=1, dtype=np.float64)
    imgs = src.images[:]

    # Generate line project angles
    ang_degrees = np.linspace(0, 180, L)
    ang_rads = ang_degrees * np.pi / 180.0

    # Call the line projection method
    s = imgs.project(ang_rads)

    # # Compare with sk
    # res = np.empty((n,L,L))
    # for i,img in enumerate(imgs.asnumpy()):
    #     #res[i] = radon(img ...)
