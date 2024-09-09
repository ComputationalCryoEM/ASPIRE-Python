import os
import sys

import numpy as np
import pytest

from aspire.basis import Coef, FBBasis2D, FLEBasis2D
from aspire.image import Image
from aspire.nufft import backend_available
from aspire.numeric import fft
from aspire.source import Simulation
from aspire.volume import Volume

from ._basis_util import UniversalBasisMixin

DATA_DIR = os.path.join(os.path.dirname(__file__), "saved_test_data")


def show_fle_params(basis):
    return f"{basis.nres}-{basis.epsilon}"


def gpu_ci_skip():
    pytest.skip("1e-7 precision for FLEBasis2D")


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
    sim = Simulation(
        L=L, n=n, vols=v, dtype=v.dtype, offsets=0, amplitudes=1, seed=1103
    )
    img = sim.clean_images[:]
    return img


def relerr(base, approx):
    # relative error of two arrays
    base = np.array(base).flatten()
    approx = np.array(approx).flatten()
    return np.linalg.norm(base - approx) / np.linalg.norm(base)


@pytest.mark.parametrize("basis", test_bases, ids=show_fle_params)
class TestFLEBasis2D(UniversalBasisMixin):
    # Loosen the tolerance for `cufinufft` to be within 15%
    test_eps = 1.15 if backend_available("cufinufft") else 1.0

    # check closeness guarantees for fast vs dense matrix method
    def testFastVDense_T(self, basis):
        if backend_available("cufinufft") and basis.epsilon == 1e-7:
            gpu_ci_skip()

        dense_b = basis._create_dense_matrix()

        # create sample particle
        x = create_images(basis.nres, 1).asnumpy()
        xvec = x.reshape((basis.nres**2, 1))

        # explicit matrix multiplication
        result_dense = dense_b.T @ xvec
        # fast evaluate_t
        result_fast = basis.evaluate_t(Image(x))

        assert relerr(result_dense.T, result_fast) < (self.test_eps * basis.epsilon)

    def testFastVDense(self, basis):
        if backend_available("cufinufft") and basis.epsilon == 1e-7:
            gpu_ci_skip()

        dense_b = basis._create_dense_matrix()

        # get sample coefficients
        x = create_images(basis.nres, 1)
        # hold input test data constant (would depend on epsilon parameter)
        coefs = FLEBasis2D(
            basis.nres, epsilon=1e-4, dtype=np.float64, match_fb=False
        ).evaluate_t(x)

        result_dense = dense_b @ coefs.asnumpy().T
        result_fast = basis.evaluate(coefs).asnumpy()

        assert relerr(result_dense, result_fast) < (self.test_eps * basis.epsilon)

    @pytest.mark.xfail(
        sys.platform == "win32",
        reason="Bug on windows with latest envs, #862",
        raises=RuntimeError,
    )
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
    coefs = Coef(basis, np.eye(basis.count))

    fb_images = fb_basis.evaluate(coefs)
    fle_images = basis.evaluate(coefs)

    np.testing.assert_allclose(fb_images._data, fle_images._data, atol=1e-4)


@pytest.mark.parametrize("basis", test_bases_match_fb, ids=show_fle_params)
def testMatchFBDenseEvaluate(basis):
    # ensure that images are the same when evaluating coefficients via slow
    # matrix multiplication

    fb_basis = FBBasis2D(basis.nres, dtype=np.float64)

    coefs = Coef(basis, np.eye(basis.count))

    fb_images = fb_basis.evaluate(coefs).asnumpy()
    fle_out = basis._create_dense_matrix() @ coefs
    fle_images = Image(fle_out.T.reshape(-1, basis.nres, basis.nres)).asnumpy()

    # Matrix column reording in match_fb mode flips signs of some of the basis functions
    np.testing.assert_allclose(np.abs(fb_images), np.abs(fle_images), atol=1e-3)
    np.testing.assert_allclose(fb_images, fle_images, atol=1e-3)


@pytest.mark.parametrize("basis", test_bases_match_fb, ids=show_fle_params)
def testMatchFBEvaluate_t(basis):
    if backend_available("cufinufft") and basis.epsilon == 1e-7:
        gpu_ci_skip()

    # ensure that coefficients are the same when evaluating images
    fb_basis = FBBasis2D(basis.nres, dtype=np.float64)

    # test images to evaluate
    images = fb_basis.evaluate(Coef(basis, np.eye(basis.count)))

    fb_coefs = fb_basis.evaluate_t(images)
    fle_coefs = basis.evaluate_t(images)

    np.testing.assert_allclose(fb_coefs, fle_coefs, atol=1e-4)


@pytest.mark.parametrize("basis", test_bases_match_fb, ids=show_fle_params)
def testMatchFBDenseEvaluate_t(basis):
    # ensure that coefficients are the same when evaluating images via slow
    # matrix multiplication

    fb_basis = FBBasis2D(basis.nres, dtype=np.float64)

    # test images to evaluate
    # gets a stack of shape (basis.count, L, L)
    images = fb_basis.evaluate(Coef(basis, np.eye(basis.count)))
    # reshape to a stack of basis.count vectors of length L**2
    vec = images.asnumpy().reshape((-1, basis.nres**2))

    fb_coefs = fb_basis.evaluate_t(images)
    fle_coefs = basis._create_dense_matrix().T @ vec.T

    # Matrix column reording in match_fb mode flips signs of some of the basis coefficients
    np.testing.assert_allclose(np.abs(fb_coefs), np.abs(fle_coefs), atol=1e-4)


def testLowPass():
    # test that low passing removes more and more high frequency
    # elements as bandlimit decreases

    L = 128
    basis = FLEBasis2D(L, match_fb=False)

    # sample coefficients
    ims = create_images(L, 1)
    coefs = basis.evaluate_t(ims)

    nonzero_coefs = []
    for i in range(4):
        bandlimit = L // (2**i)
        coefs_lowpassed = basis.lowpass(coefs, bandlimit).asnumpy()
        nonzero_coefs.append(np.sum(coefs_lowpassed != 0))

    # for bandlimit == L, no frequencies should be removed
    assert nonzero_coefs[0] == basis.count

    # for lower bandlimits, there should be fewer and fewer nonzero coefs
    assert nonzero_coefs[0] > nonzero_coefs[1] > nonzero_coefs[2] > nonzero_coefs[3]

    # make sure you can pass in a 1-D array if you want
    _ = basis.lowpass(coefs[0, :], L)


def testRadialConvolution():
    # test ability to accurately convolve with a radial
    # (e.g. CTF) function via FLE coefficients

    L = 32
    basis = FLEBasis2D(L, match_fb=False, dtype=np.float64)
    # load test radial function
    x = np.load(os.path.join(DATA_DIR, "fle_radial_fn_32x32.npy")).reshape(1, 32, 32)
    x = x / np.max(np.abs(x.flatten()))

    # get sample images
    ims = create_images(L, 10)
    # convolve using coefficients
    coefs = basis.evaluate_t(ims)
    coefs_convolved = basis.radial_convolve(coefs, x)
    imgs_convolved_fle = basis.evaluate(coefs_convolved).asnumpy()

    # convolve using FFT
    x = basis.evaluate(basis.evaluate_t(Image(x))).asnumpy()
    ims = basis.evaluate(coefs).asnumpy()

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

    np.testing.assert_allclose(imgs_convolved_fle, imgs_convolved_slow, atol=1e-5)
