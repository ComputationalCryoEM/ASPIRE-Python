import numpy as np
import pytest
from skimage import data
from skimage.transform import iradon, radon

from aspire.image import Image
from aspire.numeric import xp
from aspire.utils import grid_2d

# Relative tolerance comparing line projections to scikit
# The same tolerance will be used in all scikit forward and backward comparisons
SK_TOL_FORWARDPROJECT = 0.005

SK_TOL_BACKPROJECT = 0.2

IMG_SIZES = [
    511,
    512,
]

DTYPES = [
    np.float32,
    np.float64,
]

ANGLES = [
    1,
    50,
    pytest.param(90, marks=pytest.mark.expensive),
    pytest.param(117, marks=pytest.mark.expensive),
    pytest.param(180, marks=pytest.mark.expensive),
    pytest.param(360, marks=pytest.mark.expensive),
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


@pytest.fixture(params=ANGLES, ids=lambda x: f"n_angles={x}", scope="module")
def num_ang(request):
    """
    Number of angles in radon transform.
    """
    return request.param


@pytest.fixture
def masked_image(dtype, img_size):
    """
    Creates a masked image fixture using camera data from Scikit-Image.
    """
    g = grid_2d(img_size, normalized=True, shifted=True)
    mask = g["r"] < 1

    image = data.camera().astype(dtype)
    image = image[:img_size, :img_size]
    return Image(image * mask)


# Image.project and compare results to skimage.radon
def test_project_single(masked_image, num_ang):
    """
    Test Image.project on a single stack of images. Compares project method output with skimage project.
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
    nrms = np.sqrt(
        np.mean((s[0]._data - reference_sinogram) ** 2, axis=-1)
    ) / np.linalg.norm(reference_sinogram, axis=-1)

    np.testing.assert_array_less(
        nrms, SK_TOL_FORWARDPROJECT, "Error in image projections."
    )


def test_project_multidim(num_ang):
    """
    Test Image.project on stacks of images. Extension of test_image_project but for multi-dimensional stacks.
    """

    L = 512  # pixels
    n = 3
    m = 2

    # Generate a mask
    g = grid_2d(L, normalized=True, shifted=True)
    mask = g["r"] < 1

    # Generate images
    imgs = Image(np.random.random((m, n, L, L))) * mask

    # Generate line project angles
    angles = np.linspace(0, 360, num_ang, endpoint=False)
    rads = angles / 180.0 * np.pi
    s = imgs.project(rads)

    # Compare
    reference_sinograms = np.empty((m, n, num_ang, L))
    for i in range(m):
        for j in range(n):
            img = imgs[i, j]
            # Compute the singleton case, and compare with stack.
            single_sinogram = img.project(rads)

            # These should be allclose up to determinism in the FFT and NUFFT.
            np.testing.assert_allclose(s[i, j : j + 1], single_sinogram)

            # Next individually compute sk's radon transform for each image.
            reference_sinograms[i, j] = radon(img._data[0], theta=angles[::-1]).T

    _nrms = np.sqrt(np.mean((s - reference_sinograms) ** 2, axis=-1)) / np.linalg.norm(
        reference_sinograms, axis=-1
    )
    np.testing.assert_array_less(
        _nrms, SK_TOL_FORWARDPROJECT, "Error in image projections."
    )


def test_back_project_single(masked_image, num_ang):
    """
    Test Line.backproject on a single stack of line projections or sinogram. Compares the reconstructed image to original.
    """
    angles = xp.asarray(np.linspace(0, 360, num_ang, endpoint=False))
    angles_np = xp.asnumpy(
        angles
    )  # skimage requires numpy array while we're using cupy arrays
    rads = angles / 180 * np.pi
    sinogram = masked_image.project(rads)
    sinogram_np = sinogram.asnumpy()
    back_project = sinogram.back_project(rads)

    assert masked_image.shape == back_project.shape, "The shape must be the same."

    # generate circular mask w/ radius 1 to reconstructed image
    # aim to remove discrepencies for the edges of the image
    g = grid_2d(back_project.resolution, normalized=True, shifted=True)
    mask = g["r"] < 1
    our_back_project = back_project.asnumpy()[0] * mask

    # generating sci-kit image backproject method w/ no filter
    sk_image_iradon = iradon(sinogram_np[0].T, theta=angles_np[::-1], filter_name=None)

    # we apply a normalized root mean square error on the images to find relative error to range of ref. image
    # Note: tolerance is typically < 0.2 regardless of angles, pixels, etc.
    nrmse = np.sqrt(np.mean((our_back_project - sk_image_iradon) ** 2)) / (
        np.max(sk_image_iradon - np.min(sk_image_iradon))
    )
    assert (
        nrmse < SK_TOL_BACKPROJECT
    ), f"NRMSE is too high: {nrmse}, expected less than {SK_TOL_BACKPROJECT}"


def test_back_project_multidim(num_ang):
    """
    Test Line.backproject on a stack of images. Extension of back_project_single but for multi-dimensional stacks. Similar to forward_multidim test.
    """
    L = 512  # pixels
    n = 3
    m = 2

    g = grid_2d(L, normalized=True, shifted=True)
    mask = g["r"] < 1

    # Generate images
    imgs = Image(np.random.random((m, n, L, L))) * mask
    angles = xp.asarray(np.linspace(0, 360, num_ang, endpoint=False))
    angles_np = xp.asnumpy(angles)  # need this for the skimage transformations
    rads = angles / 180 * np.pi

    # apply a forward project on the image, then backwards
    ours_forward = imgs.project(rads)
    ours_backward = ours_forward.back_project(rads)

    # Compare
    reference_back_projects = np.empty((m, n, L, L))
    for i in range(m):
        for j in range(n):
            img = imgs[i, j]
            # Compute the singleton case, and compare with stack.
            single_sinogram = img.project(rads)
            back_project = single_sinogram.back_project(rads)

            # These should be allclose up to determinism.
            np.testing.assert_allclose(ours_backward[i, j : j + 1], back_project[0])

            # Next individually compute sk's iradon transform for each image.
            reference_back_projects[i, j] = iradon(
                single_sinogram.asnumpy()[0].T, theta=angles_np[::-1], filter_name=None
            )

    # apply a mask, then find the NRMSE on the collection of images
    # similar tolerance level to single project test
    nrmse = np.sqrt(
        np.mean((ours_backward.asnumpy() * mask - reference_back_projects) ** 2)
    ) / (np.max(reference_back_projects) - np.min(reference_back_projects))
    assert (
        nrmse < SK_TOL_BACKPROJECT
    ), f"NRMSE is too high for image ({i},{j}): {nrmse}, expected less than {SK_TOL_BACKPROJECT}"
