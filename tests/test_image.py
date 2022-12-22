import logging
import os.path

import numpy as np
import pytest
from pytest import raises
from scipy import misc

from aspire.image import Image
from aspire.utils import powerset, utest_tolerance

DATA_DIR = os.path.join(os.path.dirname(__file__), "saved_test_data")

logger = logging.getLogger(__name__)

params = [(0, np.float32), (1, np.float32), (0, np.float64), (1, np.float64)]

n = 3
mdim = 2


def get_images(parity=0, dtype=np.float32):
    size = 768 - parity
    im_np = misc.face(gray=True).astype(dtype)
    # numpy array for top-level functions that directly expect it
    im_np = misc.face(gray=True).astype(dtype)[np.newaxis, :size, :size]
    # Independent Image object for testing Image methods
    im = Image(misc.face(gray=True).astype(dtype)[:size, :size])
    return im_np, im


def get_stacks(parity=0, dtype=np.float32):
    im_np, im = get_images(parity, dtype)

    # Construct a simple stack of Images
    ims_np = np.empty((n, *im_np.shape[1:]), dtype=dtype)
    for i in range(n):
        ims_np[i] = im_np * (i + 1) / float(n)

    # Independent Image stack object for testing Image methods
    ims = Image(ims_np)

    return ims_np, ims


def get_mdim_images(parity=0, dtype=np.float32):
    ims_np, im = get_stacks(parity, dtype)
    # Multi dimensional stack Image object
    mdim = 2
    mdim_ims_np = np.concatenate([ims_np] * mdim).reshape(mdim, *ims_np.shape)
    mdim_ims = Image(mdim_ims_np)

    return mdim_ims_np, mdim_ims


def testRepr():
    # don't need to parametrize this test
    _, mdim_ims = get_mdim_images()
    r = repr(mdim_ims)
    logger.info(f"Image repr:\n{r}")


def testNonSquare():
    # don't need to parametrize this test
    """Test that an irregular Image array raises."""
    with raises(ValueError, match=r".* square .*"):
        _ = Image(np.empty((4, 5)))


@pytest.mark.parametrize("parity,dtype", params)
def testImShift(parity, dtype):
    im_np, im = get_images(parity, dtype)
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
    # ground truth numpy roll
    im3 = np.roll(im_np[0, :, :], -shifts, axis=(0, 1))

    atol = utest_tolerance(dtype)

    assert np.allclose(im0.asnumpy(), im1.asnumpy(), atol=atol)
    assert np.allclose(im1.asnumpy(), im2.asnumpy(), atol=atol)
    assert np.allclose(im0.asnumpy()[0, :, :], im3, atol=atol)


@pytest.mark.parametrize("parity,dtype", params)
def testImShiftStack(parity, dtype):
    ims_np, ims = get_stacks(parity, dtype)
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
    im3 = np.array(
        [np.roll(ims_np[i, :, :], -shifts[i], axis=(0, 1)) for i in range(n)]
    )

    atol = utest_tolerance(dtype)

    assert np.allclose(im0.asnumpy(), im1.asnumpy(), atol=atol)
    assert np.allclose(im1.asnumpy(), im2.asnumpy(), atol=atol)
    assert np.allclose(im0.asnumpy(), im3, atol=atol)


def testImageShiftErrors():
    _, im = get_images(0, np.float32)
    # test bad shift shape
    with pytest.raises(ValueError, match="Input shifts must be of shape"):
        _ = im.shift(np.array([100, 100, 100]))
    # test bad number of shifts
    with pytest.raises(ValueError, match="The number of shifts"):
        _ = im.shift(np.array([[100, 200], [100, 200]]))


@pytest.mark.parametrize("parity,dtype", params)
def testImageSqrt(parity, dtype):
    im_np, im = get_images(parity, dtype)
    ims_np, ims = get_stacks(parity, dtype)
    assert np.allclose(im.sqrt().asnumpy(), np.sqrt(im_np))
    assert np.allclose(ims.sqrt().asnumpy(), np.sqrt(ims_np))


@pytest.mark.parametrize("parity,dtype", params)
def testImageTranspose(parity, dtype):
    im_np, im = get_images(parity, dtype)
    ims_np, ims = get_stacks(parity, dtype)
    # test method and abbreviation
    assert np.allclose(im.T.asnumpy(), np.transpose(im_np, (0, 2, 1)))
    assert np.allclose(im.transpose().asnumpy(), np.transpose(im_np, (0, 2, 1)))

    # Check individual imgs in a stack
    for i in range(ims_np.shape[0]):
        assert np.allclose(ims.T[i], ims_np[i].T)
        assert np.allclose(ims.transpose()[i], ims_np[i].T)


@pytest.mark.parametrize("parity,dtype", params)
def testImageFlip(parity, dtype):
    im_np, im = get_images(parity, dtype)
    ims_np, ims = get_stacks(parity, dtype)
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


def testShape():
    ims_np, ims = get_stacks()
    assert ims.shape == ims_np.shape
    assert ims.stack_shape == ims_np.shape[:-2]
    assert ims.stack_ndim == 1


def testMultiDimShape():
    ims_np, ims = get_stacks()
    mdim_ims_np, mdim_ims = get_mdim_images()
    assert mdim_ims.shape == mdim_ims_np.shape
    assert mdim_ims.stack_shape == mdim_ims_np.shape[:-2]
    assert mdim_ims.stack_ndim == mdim
    assert mdim_ims.n_images == mdim * ims.n_images


def testBadKey():
    mdim_ims_np, mdim_ims = get_mdim_images()
    with pytest.raises(ValueError, match="slice length must be"):
        _ = mdim_ims[tuple(range(mdim_ims.ndim + 1))]


def testMultiDimGets():
    ims_np, ims = get_stacks()
    mdim_ims_np, mdim_ims = get_mdim_images()
    for X in mdim_ims:
        assert np.allclose(ims_np, X)

    # Test a slice
    assert np.allclose(mdim_ims[:, 1:], ims[1:])


def testMultiDimSets():
    ims_np, ims = get_stacks()
    mdim_ims_np, mdim_ims = get_mdim_images()
    mdim_ims[0, 1] = 123
    # Check the values changed
    assert np.allclose(mdim_ims[0, 1], 123)
    # and only those values changed
    assert np.allclose(mdim_ims[0, 0], ims_np[0])
    assert np.allclose(mdim_ims[0, 2:], ims_np[2:])
    assert np.allclose(mdim_ims[1, :], ims_np)


def testMultiDimSetsSlice():
    ims_np, ims = get_stacks()
    mdim_ims_np, mdim_ims = get_mdim_images()
    # Test setting a slice
    mdim_ims[0, 1:] = 456
    # Check the values changed
    assert np.allclose(mdim_ims[0, 1:], 456)
    # and only those values changed
    assert np.allclose(mdim_ims[0, 0], ims_np[0])
    assert np.allclose(mdim_ims[1, :], ims_np)


def testMultiDimReshape():
    # Try mdim reshape
    mdim_ims_np, mdim_ims = get_mdim_images()
    X = mdim_ims.stack_reshape(*mdim_ims.stack_shape[::-1])
    assert X.stack_shape == mdim_ims.stack_shape[::-1]
    # Compare with direct np.reshape of axes of ndarray
    shape = (*mdim_ims_np.shape[:-2][::-1], *mdim_ims_np.shape[-2:])
    assert np.allclose(X.asnumpy(), mdim_ims_np.reshape(shape))


def testMultiDimFlattens():
    mdim_ims_np, mdim_ims = get_mdim_images()
    # Try flattening
    X = mdim_ims.stack_reshape(mdim_ims.n_images)
    assert X.stack_shape, (mdim_ims.n_images,)


def testMultiDimFlattensTrick():
    mdim_ims_np, mdim_ims = get_mdim_images()
    # Try flattening with -1
    X = mdim_ims.stack_reshape(-1)
    assert X.stack_shape == (mdim_ims.n_images,)


def testMultiDimReshapeTuples():
    mdim_ims_np, mdim_ims = get_mdim_images()
    # Try flattening with (-1,)
    X = mdim_ims.stack_reshape((-1,))
    assert X.stack_shape, (mdim_ims.n_images,)

    # Try mdim reshape
    X = mdim_ims.stack_reshape(mdim_ims.stack_shape[::-1])
    assert X.stack_shape == mdim_ims.stack_shape[::-1]


def testMultiDimBadReshape():
    mdim_ims_np, mdim_ims = get_mdim_images()
    # Incorrect flat shape
    with pytest.raises(ValueError, match="Number of images"):
        _ = mdim_ims.stack_reshape(8675309)

    # Incorrect mdin shape
    with pytest.raises(ValueError, match="Number of images"):
        _ = mdim_ims.stack_reshape(42, 8675309)


def testMultiDimBroadcast():
    ims_np, ims = get_stacks()
    mdim_ims_np, mdim_ims = get_mdim_images()
    X = mdim_ims + ims
    assert np.allclose(X[0], 2 * ims.asnumpy())
