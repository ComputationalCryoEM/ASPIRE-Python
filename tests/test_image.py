import logging
import os.path
import tempfile
from datetime import datetime
from unittest import mock

import mrcfile
import numpy as np
import pytest
from PIL import Image as PILImage
from pytest import raises
from scipy.datasets import face

from aspire.image import Image, compute_fastrotate_interp_tables, fastrotate, sp_rotate
from aspire.utils import Rotation, gaussian_2d, grid_2d, powerset, utest_tolerance
from aspire.volume import CnSymmetryGroup

from .test_utils import matplotlib_dry_run

DATA_DIR = os.path.join(os.path.dirname(__file__), "saved_test_data")

logger = logging.getLogger(__name__)

params = [(0, np.float32), (1, np.float32), (0, np.float64), (1, np.float64)]

n = 3
mdim = 2

PARITY = [0, 1]
DTYPES = [np.float32, np.float64]


@pytest.fixture(params=PARITY, ids=lambda x: f"parity={x}", scope="module")
def parity(request):
    return request.param


@pytest.fixture(params=DTYPES, ids=lambda x: f"dtype={x}", scope="module")
def dtype(request):
    return request.param


@pytest.fixture(scope="module")
def get_images(parity, dtype):
    size = 768 - parity
    # numpy array for top-level functions that directly expect it
    im_np = face(gray=True).astype(dtype)[np.newaxis, :size, :size]
    denom = np.max(np.abs(im_np))
    im_np /= denom  # Normalize test image data to 0,1

    # Independent Image object for testing Image methods
    im = Image(im_np.copy(), pixel_size=1.23)
    return im_np, im


@pytest.fixture(scope="module")
def get_stacks(get_images, dtype):
    im_np, im = get_images

    # Construct a simple stack of Images
    ims_np = np.empty((n, *im_np.shape[1:]), dtype=im_np.dtype)
    for i in range(n):
        ims_np[i] = im_np * (i + 1) / float(n)

    # Independent Image stack object for testing Image methods
    ims = Image(ims_np.copy())
    return ims_np, ims


# Note that `get_mdim_images` is mutated by some tests,
# force per function scope.
@pytest.fixture(scope="function")
def get_mdim_images(get_stacks):
    ims_np, im = get_stacks
    # Multi dimensional stack Image object
    mdim = 2
    mdim_ims_np = np.concatenate([ims_np] * mdim).reshape(mdim, *ims_np.shape)

    # Independent multidimensional Image stack object for testing Image methods
    mdim_ims = Image(mdim_ims_np.copy())
    return mdim_ims_np, mdim_ims


def testRepr(get_mdim_images):
    _, mdim_ims = get_mdim_images
    r = repr(mdim_ims)
    logger.info(f"Image repr:\n{r}")


def testNonSquare():
    """Test that an irregular Image array raises."""
    with raises(ValueError, match=r".* square .*"):
        _ = Image(np.empty((4, 5)))


def testImShift(get_images, dtype):
    im_np, im = get_images
    # Note that the _im_translate method can handle float input shifts, as it
    # computes the shifts in Fourier space, rather than performing a roll
    # However, NumPy's roll() only accepts integer inputs
    shifts = np.array([100, 200])

    # test built-in
    im0 = im.shift(shifts)
    # test explicit call
    im1 = im._im_translate(shifts)
    # test that float input returns the same thing
    im2 = im.shift(shifts.astype(dtype))
    # ground truth numpy roll.
    # Note: NumPy axes 0 and 1 correspond to the row and column of an array,
    # respectively, which corresponds to the y-axis and x-axis when that array
    # represents an image. Since our shifts are (x-shifts, y-shifts), the axis
    # parameter for np.roll() must be set to (1, 0) to accomodate.
    im3 = np.roll(im_np[0, :, :], -shifts, axis=(1, 0))

    atol = utest_tolerance(dtype)

    np.testing.assert_allclose(im0.asnumpy(), im1.asnumpy(), atol=atol)
    np.testing.assert_allclose(im1.asnumpy(), im2.asnumpy(), atol=atol)
    np.testing.assert_allclose(im0.asnumpy()[0, :, :], im3, atol=atol)


def testImShiftStack(get_stacks, dtype):
    ims_np, ims = get_stacks
    # test stack of shifts (same number as Image.num_img)
    # mix of odd and even
    shifts = np.array([[100, 200], [203, 150], [55, 307]])

    # test built-in
    im0 = ims.shift(shifts)
    # test explicit call
    im1 = ims._im_translate(shifts)
    # test that float input returns the same thing
    im2 = ims.shift(shifts.astype(dtype))
    # ground truth numpy roll
    # Note: NumPy axes 0 and 1 correspond to the row and column of an array,
    # respectively, which corresponds to the y-axis and x-axis when that array
    # represents an image. Since our shifts are (x-shifts, y-shifts), the axis
    # parameter for np.roll() must be set to (1, 0) to accomodate.
    im3 = np.array(
        [np.roll(ims_np[i, :, :], -shifts[i], axis=(1, 0)) for i in range(n)]
    )

    atol = utest_tolerance(dtype)

    np.testing.assert_allclose(im0.asnumpy(), im1.asnumpy(), atol=atol)
    np.testing.assert_allclose(im1.asnumpy(), im2.asnumpy(), atol=atol)
    np.testing.assert_allclose(im0.asnumpy(), im3, atol=atol)


def testImageShiftShapeErrors():
    # Test images
    im = Image(np.ones((1, 8, 8)))
    im3 = Image(np.ones((3, 8, 8)))

    # Single image, broadcast multiple shifts is allowed
    _ = im.shift(np.array([[100, 200], [100, 200]]))

    # Multiple image, broadcast single shifts is allowed
    _ = im3.shift(np.array([[100, 200]]))

    # Bad shift shape, must be (..., 2)
    with pytest.raises(ValueError, match="Input shifts must be of shape"):
        _ = im.shift(np.array([100, 100, 100]))

    # Incoherent number of shifts (number of images != number of shifts when neither 1).
    with pytest.raises(ValueError, match="The number of shifts"):
        _ = im3.shift(np.array([[100, 200], [100, 200]]))


def testImageSqrt(get_images, get_stacks):
    im_np, im = get_images
    ims_np, ims = get_stacks
    assert np.allclose(im.sqrt().asnumpy(), np.sqrt(im_np))
    assert np.allclose(ims.sqrt().asnumpy(), np.sqrt(ims_np))


def testImageTranspose(get_images, get_stacks):
    im_np, im = get_images
    ims_np, ims = get_stacks
    # test method and abbreviation
    assert np.allclose(im.T.asnumpy(), np.transpose(im_np, (0, 2, 1)))
    assert np.allclose(im.transpose().asnumpy(), np.transpose(im_np, (0, 2, 1)))

    # Check individual imgs in a stack
    for i in range(ims_np.shape[0]):
        assert np.allclose(ims.T[i], ims_np[i].T)
        assert np.allclose(ims.transpose()[i], ims_np[i].T)


def testImageFlip(get_images, get_stacks):
    im_np, im = get_images
    ims_np, ims = get_stacks
    for axis in powerset(range(1, 3)):
        if not axis:
            # test default
            result_single = im.flip().asnumpy()
            result_stack = ims.flip().asnumpy()
            axis = 1
        else:
            result_single = im.flip(axis).asnumpy()
            result_stack = ims.flip(axis).asnumpy()
        # single image
        assert np.allclose(result_single, np.flip(im_np, axis))
        # stack
        assert np.allclose(result_stack, np.flip(ims_np, axis))

    # test error for axis 0
    axes = [0, (0, 1)]
    for axis in axes:
        with pytest.raises(ValueError, match="stack axis"):
            _ = im.flip(axis)


def testShape(get_stacks):
    ims_np, ims = get_stacks
    assert ims.shape == ims_np.shape
    assert ims.stack_shape == ims_np.shape[:-2]
    assert ims.stack_ndim == 1


def testMultiDimShape(get_stacks, get_mdim_images):
    ims_np, ims = get_stacks
    mdim_ims_np, mdim_ims = get_mdim_images
    assert mdim_ims.shape == mdim_ims_np.shape
    assert mdim_ims.stack_shape == mdim_ims_np.shape[:-2]
    assert mdim_ims.stack_ndim == mdim
    assert mdim_ims.n_images == mdim * ims.n_images


def testBadKey(get_mdim_images):
    mdim_ims_np, mdim_ims = get_mdim_images
    with pytest.raises(ValueError, match="slice length must be"):
        _ = mdim_ims[tuple(range(mdim_ims.ndim + 1))]


def testMultiDimGets(get_stacks, get_mdim_images):
    ims_np, ims = get_stacks
    mdim_ims_np, mdim_ims = get_mdim_images
    for X in mdim_ims:
        assert np.allclose(ims_np, X)

    # Test a slice
    assert np.allclose(mdim_ims[:, 1:], ims[1:])


def testMultiDimSets(get_stacks, get_mdim_images):
    ims_np, ims = get_stacks
    mdim_ims_np, mdim_ims = get_mdim_images
    mdim_ims[0, 1] = 123
    # Check the values changed
    assert np.allclose(mdim_ims[0, 1], 123)
    # and only those values changed
    assert np.allclose(mdim_ims[0, 0], ims_np[0])
    assert np.allclose(mdim_ims[0, 2:], ims_np[2:])
    assert np.allclose(mdim_ims[1, :], ims_np)


def testMultiDimSetsSlice(get_stacks, get_mdim_images):
    ims_np, ims = get_stacks
    mdim_ims_np, mdim_ims = get_mdim_images
    # Test setting a slice
    mdim_ims[0, 1:] = 456
    # Check the values changed
    assert np.allclose(mdim_ims[0, 1:], 456)
    # and only those values changed
    assert np.allclose(mdim_ims[0, 0], ims_np[0])
    assert np.allclose(mdim_ims[1, :], ims_np)


def testMultiDimReshape(get_mdim_images):
    # Try mdim reshape
    mdim_ims_np, mdim_ims = get_mdim_images
    X = mdim_ims.stack_reshape(*mdim_ims.stack_shape[::-1])
    assert X.stack_shape == mdim_ims.stack_shape[::-1]
    # Compare with direct np.reshape of axes of ndarray
    shape = (*mdim_ims_np.shape[:-2][::-1], *mdim_ims_np.shape[-2:])
    assert np.allclose(X.asnumpy(), mdim_ims_np.reshape(shape))


def testMultiDimFlattens(get_mdim_images):
    mdim_ims_np, mdim_ims = get_mdim_images
    # Try flattening
    X = mdim_ims.stack_reshape(mdim_ims.n_images)
    assert X.stack_shape, (mdim_ims.n_images,)


def testMultiDimFlattensTrick(get_mdim_images):
    mdim_ims_np, mdim_ims = get_mdim_images
    # Try flattening with -1
    X = mdim_ims.stack_reshape(-1)
    assert X.stack_shape == (mdim_ims.n_images,)


def testMultiDimReshapeTuples(get_mdim_images):
    mdim_ims_np, mdim_ims = get_mdim_images
    # Try flattening with (-1,)
    X = mdim_ims.stack_reshape((-1,))
    assert X.stack_shape, (mdim_ims.n_images,)

    # Try mdim reshape
    X = mdim_ims.stack_reshape(mdim_ims.stack_shape[::-1])
    assert X.stack_shape == mdim_ims.stack_shape[::-1]


def testMultiDimBadReshape(get_mdim_images):
    mdim_ims_np, mdim_ims = get_mdim_images
    # Incorrect flat shape
    with pytest.raises(ValueError, match="Number of images"):
        _ = mdim_ims.stack_reshape(8675309)

    # Incorrect mdin shape
    with pytest.raises(ValueError, match="Number of images"):
        _ = mdim_ims.stack_reshape(42, 8675309)


def testMultiDimBroadcast(get_stacks, get_mdim_images):
    ims_np, ims = get_stacks
    mdim_ims_np, mdim_ims = get_mdim_images
    X = mdim_ims + ims
    np.testing.assert_allclose(X[0], 2 * ims.asnumpy())


@matplotlib_dry_run
def testShow():
    """
    Test show doesn't crash.
    """
    im = Image(np.random.random((3, 8, 8)))
    im.show()


def test_backproject_symmetry_group(dtype):
    """
    Test backproject SymmetryGroup pass through and error message.
    """
    ary = np.random.random((5, 8, 8))
    im = Image(ary, dtype=dtype)
    rots = Rotation.generate_random_rotations(5).matrices

    # Attempt backproject with bad symmetry group.
    not_a_symmetry_group = []
    with raises(TypeError, match=r"`symmetry` must be a string or `SymmetryGroup`"):
        _ = im.backproject(rots, symmetry_group=not_a_symmetry_group)

    # Symmetry from string.
    vol = im.backproject(rots, symmetry_group="C3")
    assert isinstance(vol.symmetry_group, CnSymmetryGroup)

    # Symmetry from instance.
    vol = im.backproject(rots, symmetry_group=CnSymmetryGroup(order=3))
    assert isinstance(vol.symmetry_group, CnSymmetryGroup)


def test_asnumpy_readonly():
    """
    Attempting assignment should raise an error.
    """
    ary = np.random.random((3, 8, 8))
    im = Image(ary)
    vw = im.asnumpy()

    # Attempt assignment
    with raises(ValueError, match=r".*destination is read-only.*"):
        vw[0, 0, 0] = 123


def test_save_overwrite(caplog):
    """
    Test that the overwrite flag behaves as expected.
    - overwrite=True: Overwrites the existing file.
    - overwrite=False: Raises an error if the file exists.
    - overwrite=None: Renames the existing file and saves the new one.
    """
    im1 = Image(np.ones((1, 8, 8), dtype=np.float32))
    im2 = Image(2 * np.ones((1, 8, 8), dtype=np.float32))
    im3 = Image(3 * np.ones((1, 8, 8), dtype=np.float32))

    # Create a tmp dir for this test output
    with tempfile.TemporaryDirectory() as tmpdir_name:
        # tmp filename
        mrc_path = os.path.join(tmpdir_name, "og.mrc")
        base, ext = os.path.splitext(mrc_path)

        # Create and save the first image
        im1.save(mrc_path, overwrite=True)

        # Case 1: overwrite=True (should overwrite the existing file)
        im2.save(mrc_path, overwrite=True)

        # Load and check if im2 has overwritten im1
        im2_loaded = Image.load(mrc_path)
        np.testing.assert_allclose(im2.asnumpy(), im2_loaded.asnumpy())

        # Case 2: overwrite=False (should raise an overwrite error)
        with pytest.raises(
            ValueError,
            match="File '.*' already exists; set overwrite=True to overwrite it",
        ):
            im3.save(mrc_path, overwrite=False)

        # Case 3: overwrite=None (should rename the existing file and save im3 with original filename)
        # Mock datetime to return a fixed timestamp.
        mock_datetime_value = datetime(2024, 10, 18, 12, 0, 0)
        with mock.patch("aspire.utils.misc.datetime") as mock_datetime:
            mock_datetime.now.return_value = mock_datetime_value
            mock_datetime.strftime = datetime.strftime

            with caplog.at_level(logging.INFO):
                im3.save(mrc_path, overwrite=None)

                # Check that the existing file was renamed and logged
                assert f"Renaming {mrc_path}" in caplog.text

                # Construct the expected renamed filename using the mock timestamp
                mock_timestamp = mock_datetime_value.strftime("%y%m%d_%H%M%S")
                renamed_file = f"{base}_{mock_timestamp}{ext}"

                # Assert that the renamed file exists
                assert os.path.exists(renamed_file), "Renamed file not found"

        # Load and check that im3 was saved to the original path
        im3_loaded = Image.load(mrc_path)
        np.testing.assert_allclose(im3.asnumpy(), im3_loaded.asnumpy())

        # Also check that the renamed file still contains im2's data
        im2_loaded_renamed = Image.load(renamed_file)
        np.testing.assert_allclose(im2.asnumpy(), im2_loaded_renamed.asnumpy())


def test_corrupt_mrc_load(caplog):
    """
    Test that corrupt mrc files are logged as expected.
    """

    # Create a tmp dir for this test output
    with tempfile.TemporaryDirectory() as tmpdir_name:
        # tmp filename
        mrc_path = os.path.join(tmpdir_name, "bad.mrc")

        # Create and save image
        Image(np.ones((1, 8, 8), dtype=np.float32)).save(mrc_path)

        # Open mrc file and soft corrupt it
        with mrcfile.open(mrc_path, "r+") as fh:
            fh.header.map = -1

        # Check that we get a WARNING
        with caplog.at_level(logging.WARNING):
            _ = Image.load(mrc_path)

            # Check the message prefix
            assert f"Image.load of {mrc_path} reporting 1 corruptions" in caplog.text

            # Check the message contains the file path
            assert mrc_path in caplog.text

            caplog.clear()


def test_load_bad_ext():
    """
    Check error raised when attempting to load unsupported file.
    """
    with raises(RuntimeError, match=r".*unsupported file extension.*"):
        _ = Image.load("bad.ext")


def test_load_mrc(dtype):
    """
    Test `Image.load` round-trip for `mrc` extension.
    """

    # `sample.mrc` is single precision
    filepath = os.path.join(DATA_DIR, "sample.mrc")

    # Load data from file
    im = Image.load(filepath, dtype=dtype)

    with tempfile.TemporaryDirectory() as tmpdir_name:
        # tmp filename
        test_filepath = os.path.join(tmpdir_name, "test.mrc")

        im.save(test_filepath)

        im2 = Image.load(test_filepath, dtype)

    # Check the single precision round-trip
    assert np.array_equal(im, im2)
    assert im2.dtype == dtype


def test_load_mrcs(dtype):
    """
    Test `Image.load` round-trip for `mrcs` extension.
    """

    # `sample.mrcs` is single precision
    filepath = os.path.join(DATA_DIR, "sample.mrcs")

    # Load data from file
    im = Image.load(filepath, dtype=dtype)

    with tempfile.TemporaryDirectory() as tmpdir_name:
        # tmp filename
        test_filepath = os.path.join(tmpdir_name, "test.mrcs")

        im.save(test_filepath)

        im2 = Image.load(test_filepath, dtype)

    # Check the single precision round-trip
    assert np.array_equal(im, im2)
    assert im2.dtype == dtype


def test_load_tiff():
    """
    Test `Image.load` with a TIFF file
    """

    # `sample.mrc` is single precision
    filepath = os.path.join(DATA_DIR, "sample.mrc")

    # Load data from file
    im = Image.load(filepath)

    with tempfile.TemporaryDirectory() as tmpdir_name:
        # tmp filename
        test_filepath = os.path.join(tmpdir_name, "test.tiff")

        # Write image data as TIFF
        PILImage.fromarray(im.asnumpy()[0]).save(test_filepath)

        # Load TIFF into Image
        im2 = Image.load(test_filepath)

    # Check contents
    assert np.array_equal(im, im2)


def test_save_load_pixel_size(get_images, dtype):
    """
    Test saving and loading an MRC with pixel size attribute
    """

    im_np, im = get_images

    with tempfile.TemporaryDirectory() as tmpdir_name:
        # tmp filename
        test_filepath = os.path.join(tmpdir_name, "test.mrc")

        # Save image to file
        im.save(test_filepath)

        # Load image from file
        im2 = Image.load(test_filepath, dtype)

    # Check we've loaded the image data
    np.testing.assert_allclose(im2, im)
    # Check we've loaded the image dtype
    assert im2.dtype == im.dtype, "Image dtype mismatched on save-load"
    # Check we've loaded the pixel size
    np.testing.assert_almost_equal(
        im2.pixel_size, im.pixel_size, err_msg="Image pixel_size incorrect save-load"
    )


@pytest.fixture(
    params=Image.rotation_methods, ids=lambda x: f"method={x}", scope="module"
)
def rotation_method(request):
    return request.param


def test_image_rotate(dtype, rotation_method):
    """
    Compare image rotations against rotated gaussian blobs.
    """

    L = 129  # Test image size in pixels
    num_test_angles = 42
    # Create mask, used to zero edge artifacts
    mask = grid_2d(L, normalized=True)["r"] < 0.9

    def _gen_image(angle, L, n=1, K=10):
        """
        Generate `n` `L-by-L` image arrays,
        each constructed by a sequence of `K` gaussian blobs,
        and reference images with the blob centers rotated by `angle`.

        Return tuple of unrotated and rotated image arrays (n-by-L-by-L).

        :param angle: rotation angle
        :param L: size (L-by-L) in pixels
        :param K: Number of blobs
        :return:
            - Array of unrotated data (float64)
            - Array of rotated data (float64)
        """

        im = np.zeros((n, L, L), dtype=np.float64)
        rotated_im = np.zeros_like(im)

        centers = np.random.randint(-L // 4, L // 4, size=(n, 10, 2))
        sigmas = np.full((n, K, 2), L / 10, dtype=np.float64)

        # Rotate the gaussian specifications
        R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        rotated_centers = centers @ R

        # Construct each image independently
        for i in range(n):
            for center, sigma in zip(centers[i], sigmas[i]):
                im[i] = im[i] + gaussian_2d(L, center, sigma, dtype=np.float64)

            for center, sigma in zip(rotated_centers[i], sigmas[i]):
                rotated_im[i] = rotated_im[i] + gaussian_2d(
                    L, center, sigma, dtype=np.float64
                )

        return im, rotated_im

    # Test over a variety of angles `theta`
    for theta in np.linspace(0, 2 * np.pi, num_test_angles):
        # Generate images and reference (`theta`) rotated images
        im, ref = _gen_image(theta, L, n=3)
        im = Image(im.astype(dtype, copy=False))

        # Rotate using `Image`'s `rotation_method`
        im_rot = im.rotate(theta, method=rotation_method)

        # Mask off boundary artifacts
        masked_diff = (im_rot - ref) * mask

        # Compute L1 error of masked diff, per image
        L1_error = np.mean(np.abs(masked_diff), axis=(-1, -2))
        np.testing.assert_array_less(
            L1_error,
            0.1,
            err_msg=f"{L} pixels using {rotation_method} @ {theta} radians",
        )


def test_sp_rotate_inputs(dtype):
    """
    Smoke test various input combinations to the scipy rotation wrapper.
    """

    imgs = np.zeros((6, 8, 8), dtype=dtype)
    thetas = np.arange(6, dtype=dtype)
    theta = thetas[0]  # scalar

    # #  These are the only supported calls admitted by the function doc.
    # singleton, scalar
    _ = sp_rotate(imgs[0], theta)
    # stack, scalar
    _ = sp_rotate(imgs, theta)

    # #  These happen to also work with the code, so were put under test.
    # #    We're not advertising them, as there really isn't a good use
    # #    case for this wrapper code outside of the internal wrapping
    # #    application.
    # singleton, single element array
    _ = sp_rotate(imgs[0], thetas[0:1])
    # stack, single element array
    _ = sp_rotate(imgs, thetas[0:1])
    # stack, stack
    _ = sp_rotate(imgs, thetas)
    # md-stack, md-stack
    _ = sp_rotate(imgs.reshape(2, 3, 8, 8), thetas.reshape(2, 3, 1))
    _ = sp_rotate(imgs.reshape(2, 3, 8, 8), thetas.reshape(2, 3))


def test_fastrotate_inputs(dtype):
    """
    Smoke test various input combinations to `fastrotate`.
    """

    imgs = np.zeros((6, 8, 8), dtype=dtype)
    theta = 42

    # #  These are the supported calls
    # singleton, scalar
    _ = fastrotate(imgs[0], theta)
    # stack, scalar
    _ = fastrotate(imgs, theta)

    # #  These can also remain under test, but are not advertised.
    # stack, single element array
    _ = fastrotate(imgs, np.array(theta))
    # singleton, single element array
    _ = fastrotate(imgs[0], np.array(theta))


def test_fastrotate_M_arg(dtype):
    """
    Smoke test precomputed `M` input  to `fastrotate`.
    """

    imgs = np.random.randn(6, 8, 8).astype(dtype)
    theta = np.random.uniform(0, 2 * np.pi)

    # Precompute M
    M = compute_fastrotate_interp_tables(theta, *imgs.shape[-2:])

    # Call with theta None
    im_rot_M = fastrotate(imgs, None, M=M)
    # Compare to calling withou `M`
    im_rot = fastrotate(imgs, theta)
    np.testing.assert_allclose(im_rot_M, im_rot)

    # Call with theta, should raise
    with raises(RuntimeError, match=r".*`theta` must be `None`.*"):
        _ = fastrotate(imgs, theta, M=M)
