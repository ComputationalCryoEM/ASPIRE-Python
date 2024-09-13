import numpy as np
import pytest
from skimage import data
from skimage.transform import iradon, radon

from aspire.image import Image
from aspire.utils import grid_2d

# Relative tolerance comparing sinogram projections to scikit
# The same tolerance will be used in all scikit forward and backward comparisons
SK_TOL_FORWARDPROJECT = 0.005

SK_TOL_BACKPROJECT = 0.0025

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
    mask = g["r"] < 0.99

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
    # Note, `radon` does not admit read only views, so the slice is copied.
    reference_sinogram = radon(masked_image.asnumpy()[0].copy(), theta=angles[::-1]).T
    assert reference_sinogram.shape == (len(angles), ny), "Incorrect Shape"

    # compare project method on ski-image reference
    nrms = np.sqrt(
        np.mean((s[0].asnumpy() - reference_sinogram) ** 2, axis=-1)
    ) / np.linalg.norm(reference_sinogram, axis=-1)

    np.testing.assert_array_less(
        nrms, SK_TOL_FORWARDPROJECT, err_msg="Error in image projections."
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
    mask = g["r"] < 0.99

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
            # Note, `radon` does not admit read only views, so the slice is copied.
            reference_sinograms[i, j] = radon(
                img.asnumpy()[0].copy(), theta=angles[::-1]
            ).T

    _nrms = np.sqrt(np.mean((s - reference_sinograms) ** 2, axis=-1)) / np.linalg.norm(
        reference_sinograms, axis=-1
    )
    np.testing.assert_array_less(
        _nrms, SK_TOL_FORWARDPROJECT, err_msg="Error in image projections."
    )


def test_backproject_single(masked_image, num_ang):
    """
    Test Sinogram.backproject on a single stack of line projections (sinograms).

    This test compares the reconstructed image from the `backproject` method to
    the skimage method `iradon.`
    """
    angles = np.linspace(0, 360, num_ang, endpoint=False)
    rads = angles / 180 * np.pi
    sinogram = masked_image.project(rads)
    sinogram_np = sinogram.asnumpy()
    back_project = sinogram.backproject(rads)

    assert masked_image.shape == back_project.shape, "The shape must be the same."

    # generate circular mask w/ radius 1 to reconstructed image
    # aim to remove discrepencies for the edges of the image
    g = grid_2d(back_project.resolution, normalized=True, shifted=True)
    mask = g["r"] < 0.99
    our_back_project = back_project.asnumpy()[0] * mask

    # generating sci-kit image backproject method w/ no filter
    sk_image_iradon = iradon(sinogram_np[0].T, theta=-angles, filter_name=None) * mask

    # we apply a normalized root mean square error on the images to find relative error to range of ref. image
    nrmse = np.sqrt(np.mean((our_back_project - sk_image_iradon) ** 2)) / (
        np.max(sk_image_iradon) - np.min(sk_image_iradon)
    )
    np.testing.assert_array_less(
        nrmse,
        SK_TOL_BACKPROJECT,
        err_msg=f"NRMSE is too high: {nrmse}, expected less than {SK_TOL_BACKPROJECT}",
    )


def test_backproject_multidim(num_ang):
    """
    Test Sinogram.backproject on a stack of line projections.

    Extension of the `backproject_single` test but checks for multi-dimensional stacks.
    """
    L = 512  # pixels
    n = 3
    m = 2

    g = grid_2d(L, normalized=True, shifted=True)
    mask = g["r"] < 0.99

    # Generate images
    imgs = Image(np.random.random((m, n, L, L))) * mask
    angles = np.linspace(0, 360, num_ang, endpoint=False)
    rads = angles / 180 * np.pi

    # apply a forward project on the image, then backwards
    ours_forward = imgs.project(rads)
    ours_backward = ours_forward.backproject(rads)

    # Compare
    reference_back_projects = np.empty((m, n, L, L))
    for i in range(m):
        for j in range(n):
            img = imgs[i, j]
            # Compute the singleton case, and compare with stack.
            single_sinogram = img.project(rads)
            back_project = single_sinogram.backproject(rads)

            # These should be allclose up to determinism.
            np.testing.assert_allclose(ours_backward[i, j : j + 1], back_project[0])

            # Next individually compute sk's iradon transform for each image.
            reference_back_projects[i, j] = (
                iradon(
                    single_sinogram.asnumpy()[0].T, theta=-1 * angles, filter_name=None
                )
                * mask
            )

            # apply a mask, then find the NRMSE on the collection of images
            # similar tolerance level to single project test
    nrmse = np.sqrt(
        np.mean(
            (ours_backward.asnumpy() * mask - reference_back_projects), axis=(-2, -1)
        )
        ** 2
    ) / (
        np.max(reference_back_projects, axis=(-2, -1))
        - np.min(reference_back_projects, axis=(-2, -1))
    )

    np.testing.assert_array_less(
        nrmse, SK_TOL_BACKPROJECT, err_msg="Error with the reconstructed images."
    )


# testing the str method
def test_sinogram_str_method(masked_image, num_ang):
    angles = np.linspace(0, 360, num_ang, endpoint=False)
    rads = angles / 180 * np.pi
    sinogram = masked_image.project(rads)
    n_images = sinogram.n
    n_angles = sinogram.n_angles
    n_radial_points = sinogram.n_radial_points
    expected_str = f"Sinogram(n_images = {n_images}, n_angles = {n_angles}, n_radial_points = {n_radial_points})"
    assert str(sinogram) == expected_str


# testing the repr method
def test_sinogram_repr_method(masked_image, num_ang):
    angles = np.linspace(0, 360, num_ang, endpoint=False)
    rads = angles / 180 * np.pi
    sinogram = masked_image.project(rads)
    n_images = sinogram.n
    dtype = sinogram.dtype
    stack_shape = sinogram.stack_shape
    n_angles = sinogram.n_angles
    n_radial_points = sinogram.n_radial_points
    expected_repr = (
        f"Sinogram: {n_images} images of dtype {dtype}, "
        f"arranged as a stack with shape {stack_shape}. "
        f"Each image has {n_angles} angles and {n_radial_points} radial points."
    )
    assert repr(sinogram) == expected_repr
