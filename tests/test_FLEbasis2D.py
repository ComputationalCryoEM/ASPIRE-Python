import os

import numpy as np
import pytest

from aspire.basis import FBBasis2D, FLEBasis2D
from aspire.image import Image
from aspire.nufft import backend_available
from aspire.numeric import fft
from aspire.source import Simulation
from aspire.utils import utest_tolerance
from aspire.volume import Volume

from ._basis_util import UniversalBasisMixin

DATA_DIR = os.path.join(os.path.dirname(__file__), "saved_test_data")


def show_fle_params(basis):
    return f"{basis.nres}-{basis.epsilon}"


def gpu_ci_skip():
    pytest.skip("1e-7 precision for FLEBasis2D.evaluate()")


fle_params = [
    (32, 1e-4),
    (32, 1e-7),
    (32, 1e-10),
    (32, 1e-14),
    (33, 1e-4),
    (33, 1e-7),
    (33, 1e-10),
    (33, 1e-14),
]

test_bases = [
    FLEBasis2D(L, epsilon=epsilon, dtype=np.float64, match_fb=False)
    for L, epsilon in fle_params
]

# add one case ensuring input/output dtypes for evaluate and evaluate_t
test_bases.append(FLEBasis2D(8, epsilon=1e-4, dtype=np.float32, match_fb=False))

test_bases_match_fb = [
    FLEBasis2D(L, epsilon=epsilon, dtype=np.float64) for L, epsilon in fle_params
]


def create_images(L, n):
    # create sample data
    v = Volume(
        np.load(os.path.join(DATA_DIR, "clean70SRibosome_vol.npy")).astype(np.float64)
    )
    v = v.downsample(L)
    sim = Simulation(L=L, n=n, vols=v, dtype=v.dtype, seed=1103)
    img = sim.clean_images[:]
    return img


def relerr(base, approx):
    # relative error of two arrays
    base = np.array(base).flatten()
    approx = np.array(approx).flatten()
    return np.linalg.norm(base - approx) / np.linalg.norm(base)


@pytest.mark.parametrize("basis", test_bases, ids=show_fle_params)
class TestFLEBasis2D(UniversalBasisMixin):
    # check closeness guarantees for fast vs dense matrix method
    def testFastVDense_T(self, basis):
        dense_b = basis._create_dense_matrix()

        # create sample particle
        x = create_images(basis.nres, 1).asnumpy()
        xvec = x.reshape((basis.nres**2, 1))

        # explicit matrix multiplication
        result_dense = dense_b.T @ xvec
        # fast evaluate_t
        result_fast = basis.evaluate_t(Image(x))

        assert relerr(result_dense.T, result_fast) < basis.epsilon

    def testFastVDense(self, basis):
        if backend_available("cufinufft") and basis.epsilon == 1e-7:
            gpu_ci_skip()

        dense_b = basis._create_dense_matrix()

        # get sample coefficients
        x = create_images(basis.nres, 1)
        # hold input test data constant (would depend on epsilon parameter)
        coeffs = FLEBasis2D(
            basis.nres, epsilon=1e-4, dtype=np.float64, match_fb=False
        ).evaluate_t(x)

        result_dense = dense_b @ coeffs.T
        result_fast = basis.evaluate(coeffs).asnumpy()

        assert relerr(result_dense, result_fast) < basis.epsilon

    def testEvaluateExpand(self, basis):
        if backend_available("cufinufft") and basis.epsilon == 1e-7:
            gpu_ci_skip()

        # compare result of evaluate() vs more accurate expand()
        # get sample coefficients
        x = create_images(basis.nres, 1)
        # hold input test data constant (would depend on epsilon parameter)
        evaluate_t = basis.evaluate(basis.evaluate_t(x))
        expand = basis.evaluate(basis.expand(evaluate_t))

        assert relerr(expand.asnumpy(), evaluate_t.asnumpy()) < basis.epsilon


@pytest.mark.parametrize("basis", test_bases_match_fb, ids=show_fle_params)
def testMatchFBEvaluate(basis):
    if backend_available("cufinufft") and basis.epsilon == 1e-7:
        gpu_ci_skip()

    # ensure that the basis functions are identical when in match_fb mode
    fb_basis = FBBasis2D(basis.nres, dtype=np.float64)

    # in match_fb, count is the same for both bases
    coeffs = np.eye(basis.count)

    fb_images = fb_basis.evaluate(coeffs)
    fle_images = basis.evaluate(coeffs)

    assert np.allclose(fb_images._data, fle_images._data, atol=1e-4)


@pytest.mark.parametrize("basis", test_bases_match_fb, ids=show_fle_params)
def testMatchFBDenseEvaluate(basis):
    # ensure that images are the same when evaluating coefficients via slow
    # matrix multiplication

    fb_basis = FBBasis2D(basis.nres, dtype=np.float64)

    coeffs = np.eye(basis.count)

    fb_images = fb_basis.evaluate(coeffs).asnumpy()
    fle_out = basis._create_dense_matrix() @ coeffs
    fle_images = Image(fle_out.T.reshape(-1, basis.nres, basis.nres)).asnumpy()

    # odd resolution has to be normalized
    if basis.nres % 2 == 1:
        fb_images = fb_images / np.max(np.abs(fb_images))
        fle_images = fle_images / np.max(np.abs(fle_images))

    # Matrix column reording in match_fb mode flips signs of some of the basis functions
    assert np.allclose(np.abs(fb_images), np.abs(fle_images), atol=1e-3)


@pytest.mark.parametrize("basis", test_bases_match_fb, ids=show_fle_params)
def testMatchFBEvaluate_t(basis):
    # ensure that coefficients are the same when evaluating images

    # see #738
    if basis.nres % 2 == 1:
        odd_resolution_skip()

    fb_basis = FBBasis2D(basis.nres, dtype=np.float64)

    # test images to evaluate
    images = fb_basis.evaluate(np.eye(basis.count))

    fb_coeffs = fb_basis.evaluate_t(images)
    fle_coeffs = basis.evaluate_t(images)

    assert np.allclose(fb_coeffs, fle_coeffs, atol=1e-4)


@pytest.mark.parametrize("basis", test_bases_match_fb, ids=show_fle_params)
def testMatchFBDenseEvaluate_t(basis):
    # ensure that coefficients are the same when evaluating images via slow
    # matrix multiplication

    fb_basis = FBBasis2D(basis.nres, dtype=np.float64)

    # test images to evaluate
    # gets a stack of shape (basis.count, L, L)
    images = fb_basis.evaluate(np.eye(basis.count))
    # reshape to a stack of basis.count vectors of length L**2
    vec = images.asnumpy().reshape((-1, basis.nres**2))

    fb_coeffs = fb_basis.evaluate_t(images)
    fle_coeffs = basis._create_dense_matrix().T @ vec.T

    # odd resolution has to be normalized
    if basis.nres % 2 == 1:
        fb_coeffs = fb_coeffs / np.max(np.abs(fb_coeffs))
        fle_coeffs = fle_coeffs / np.max(np.abs(fle_coeffs))

    # Matrix column reording in match_fb mode flips signs of some of the basis coefficients
    assert np.allclose(np.abs(fb_coeffs), np.abs(fle_coeffs), atol=1e-4)


def testLowPass():
    # test that low passing removes more and more high frequency
    # elements as bandlimit decreases

    L = 128
    basis = FLEBasis2D(L, match_fb=False)

    # sample coefficients
    ims = create_images(L, 1)
    coeffs = basis.evaluate_t(ims)

    nonzero_coeffs = []
    for i in range(4):
        bandlimit = L // (2**i)
        coeffs_lowpassed = basis.lowpass(coeffs, bandlimit)
        nonzero_coeffs.append(np.sum(coeffs_lowpassed != 0))

    # for bandlimit == L, no frequencies should be removed
    assert nonzero_coeffs[0] == basis.count

    # for lower bandlimits, there should be fewer and fewer nonzero coeffs
    assert nonzero_coeffs[0] > nonzero_coeffs[1] > nonzero_coeffs[2] > nonzero_coeffs[3]

    # make sure you can pass in a 1-D array if you want
    _ = basis.lowpass(coeffs[0, :], L)

    # cannot pass in the wrong number of coefficients
    with pytest.raises(
        AssertionError, match="Number of coefficients must match self.count."
    ):
        _ = basis.lowpass(coeffs[:, :1000], L)

    # cannot pass in wrong shape
    with pytest.raises(
        AssertionError,
        match="Input a stack of coefficients of dimension",
    ):
        _ = basis.lowpass(np.zeros((3, 3, 3)), L)


def testRotate():
    # test ability to accurately rotate images via
    # FLE coefficients

    L = 128
    basis = FLEBasis2D(L, match_fb=False)

    # sample image
    ims = create_images(L, 1)
    # rotate 90 degrees in cartesian coordinates
    ims_90 = Image(np.rot90(ims.asnumpy(), axes=(1, 2)))

    # get FLE coefficients
    coeffs = basis.evaluate_t(ims)
    coeffs_cart_rot = basis.evaluate_t(ims_90)

    # rotate original image in FLE space using Steerable rotate method
    coeffs_fle_rot = basis.rotate(coeffs, np.pi / 2)

    # back to cartesian
    ims_cart_rot = basis.evaluate(coeffs_cart_rot)
    ims_fle_rot = basis.evaluate(coeffs_fle_rot)

    # test rot90 close
    assert np.allclose(ims_cart_rot[0], ims_fle_rot[0], atol=1e-4)

    # 2Pi identity in FLE space (rotate by 2Pi)
    coeffs_fle_2pi = basis.rotate(coeffs, 2 * np.pi)
    ims_fle_2pi = basis.evaluate(coeffs_fle_2pi)

    # test 2Pi identity
    assert np.allclose(ims[0], ims_fle_2pi[0], atol=utest_tolerance(basis.dtype))

    # Reflect in FLE space (rotate by Pi)
    coeffs_fle_pi = basis.rotate(coeffs, np.pi)
    ims_fle_pi = basis.evaluate(coeffs_fle_pi)

    # test reflection
    assert np.allclose(np.flipud(ims[0]), ims_fle_pi[0], atol=1e-4)

    # make sure you can pass in a 1-D array if you want
    _ = basis.lowpass(np.zeros((basis.count,)), np.pi)

    # cannot pass in the wrong number of coefficients
    with pytest.raises(
        AssertionError, match="Number of coefficients must match self.count."
    ):
        _ = basis.rotate(np.zeros((1, 10)), np.pi)

    # cannot pass in wrong shape
    with pytest.raises(
        AssertionError,
        match="Input a stack of coefficients of dimension",
    ):
        _ = basis.lowpass(np.zeros((3, 3, 3)), np.pi)


def testRadialConvolution():
    # test ability to accurately convolve with a radial
    # (e.g. CTF) function via FLE coefficients

    L = 32
    basis = FLEBasis2D(L, match_fb=False)
    # load test radial function
    x = np.load(os.path.join(DATA_DIR, "fle_radial_fn_32x32.npy")).reshape(1, 32, 32)
    x = x / np.max(np.abs(x.flatten()))

    # get sample images
    ims = create_images(L, 10)
    # convolve using coefficients
    coeffs = basis.evaluate_t(ims)
    coeffs_convolved = basis.radial_convolve(coeffs, x)
    imgs_convolved_fle = basis.evaluate(coeffs_convolved).asnumpy()

    # convolve using FFT
    x = basis.evaluate(basis.evaluate_t(x)).asnumpy()
    ims = basis.evaluate(coeffs).asnumpy()

    imgs_convolved_slow = np.zeros((10, L, L))
    for i in range(10):
        x_pad = np.zeros((2 * L, 2 * L))
        ims_pad = np.zeros((2 * L, 2 * L))
        x_pad[L // 2 : L // 2 + L, L // 2 : L // 2 + L] = x[0, :, :]
        ims_pad[L // 2 : L // 2 + L, L // 2 : L // 2 + L] = ims[i, :, :]

        x_shift = fft.fftshift(x_pad.reshape(2 * L, 2 * L))
        ims_shift = fft.fftshift(ims_pad.reshape(2 * L, 2 * L))

        convolution_fft_pad = fft.fftshift(
            fft.ifft2(np.fft.fft2(x_shift) * np.fft.fft2(ims_shift))
        )
        imgs_convolved_slow[i, :, :] = np.real(
            convolution_fft_pad[L // 2 : L // 2 + L, L // 2 : L // 2 + L]
        )

    assert np.allclose(imgs_convolved_fle, imgs_convolved_slow, atol=1e-5)
