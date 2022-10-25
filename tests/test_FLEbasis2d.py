import os
from unittest import TestCase

import numpy as np
from parameterized import parameterized

from aspire.basis import FLEBasis2D
from aspire.image import Image
from aspire.numeric import fft
from aspire.source import Simulation
from aspire.utils import utest_tolerance
from aspire.volume import Volume

from ._basis_util import UniversalBasisMixin

DATA_DIR = os.path.join(os.path.dirname(__file__), "saved_test_data")


class FLEBasis2DTestCase(TestCase, UniversalBasisMixin):
    L = 8
    dtype = np.float32

    def setUp(self):
        self.basis = FLEBasis2D((self.L, self.L), dtype=self.dtype)

    # even and odd images with all guaranteed epsilons from paper
    @parameterized.expand(
        [
            [32, 1e-4],
            [32, 1e-7],
            [32, 1e-10],
            [32, 1e-14],
            [33, 1e-4],
            [33, 1e-7],
            [33, 1e-10],
            [33, 1e-14],
        ]
    )
    # check closeness guarantees for fast vs dense matrix method
    def testFastVDense_T(self, L, epsilon):
        basis = FLEBasis2D(L, epsilon=epsilon, dtype=np.float64)
        dense_b = basis.create_dense_matrix()

        # create sample particle
        x = self.create_images(L, 1).asnumpy()
        xvec = x.reshape((L**2, 1))

        # explicit matrix multiplication
        result_dense = dense_b.T @ xvec
        # fast evaluate_t
        result_fast = basis.evaluate_t(Image(x))

        relerr = self.relerr(result_dense.T, result_fast)
        self.assertTrue(relerr < epsilon)

    # even and odd images with all guaranteed epsilons from paper
    @parameterized.expand(
        [
            [32, 1e-4],
            [32, 1e-7],
            [32, 1e-10],
            [32, 1e-14],
            [33, 1e-4],
            [33, 1e-7],
            [33, 1e-10],
            [33, 1e-14],
        ]
    )
    def testFastVDense(self, L, epsilon):
        basis = FLEBasis2D(L, epsilon=epsilon, dtype=np.float64)
        dense_b = basis.create_dense_matrix()

        # get sample coefficients
        x = self.create_images(L, 1)
        # hold input test data constant (would depend on epsilon parameter)
        coeffs = FLEBasis2D(L, epsilon=1e-4, dtype=np.float64).evaluate_t(x)

        result_dense = dense_b @ coeffs.T
        result_fast = basis.evaluate(coeffs).asnumpy()

        relerr = self.relerr(result_dense, result_fast)
        self.assertTrue(relerr < epsilon)

    @parameterized.expand(
        [
            [32, 1e-4],
            [32, 1e-7],
            [32, 1e-10],
            [32, 1e-14],
        ]
    )
    def testEvaluateExpand(self, L, epsilon):
        # compare result of evaluate() vs more accurate expand()
        basis = FLEBasis2D(L, epsilon=epsilon, dtype=np.float64)
        # get sample coefficients
        x = self.create_images(L, 1)
        # hold input test data constant (would depend on epsilon parameter)
        evaluate_t = basis.evaluate(basis.evaluate_t(x))
        expand = basis.evaluate(basis.expand(evaluate_t))

        relerr = self.relerr(evaluate_t.asnumpy(), expand.asnumpy())
        self.assertTrue(relerr < epsilon)

    def testLowPass(self):
        # test that low passing removes more and more high frequency
        # elements as bandlimit decreases

        L = 128
        basis = FLEBasis2D(L)

        # sample coefficients
        ims = self.create_images(L, 1)
        coeffs = basis.evaluate_t(ims)

        nonzero_coeffs = []
        for i in range(4):
            bandlimit = L // (2**i)
            coeffs_lowpassed = basis.lowpass(coeffs, bandlimit)
            nonzero_coeffs.append(np.sum(coeffs_lowpassed != 0))

        # for bandlimit == L, no frequencies should be removed
        self.assertEqual(nonzero_coeffs[0], basis.count)

        # for lower bandlimits, there should be fewer and fewer nonzero coeffs
        self.assertTrue(
            nonzero_coeffs[0]
            > nonzero_coeffs[1]
            > nonzero_coeffs[2]
            > nonzero_coeffs[3]
        )

        # make sure you can pass in a 1-D array if you want
        _ = basis.lowpass(coeffs[0, :], L)

        # cannot pass in the wrong number of coefficients
        with self.assertRaisesRegex(
            AssertionError, "Number of coefficients must match self.count."
        ):
            _ = basis.lowpass(coeffs[:, :1000], L)

        # cannot pass in wrong shape
        with self.assertRaisesRegex(
            AssertionError,
            "Input a stack of coefficients of dimension",
        ):
            _ = basis.lowpass(np.zeros((3, 3, 3)), L)

    def testRotate(self):
        # test ability to accurately rotate images via
        # FLE coefficients

        L = 128
        basis = FLEBasis2D(L)

        # sample image
        ims = self.create_images(L, 1)
        # rotate 90 degrees in cartesian coordinates
        ims_90 = Image(np.rot90(ims.asnumpy(), axes=(1, 2)))

        # get FLE coefficients
        coeffs = basis.evaluate_t(ims)
        coeffs_cart_rot = basis.evaluate_t(ims_90)

        # rotate original image in FLE space
        coeffs_fle_rot = basis.rotate(coeffs, np.pi / 2)

        # back to cartesian
        ims_cart_rot = basis.evaluate(coeffs_cart_rot)
        ims_fle_rot = basis.evaluate(coeffs_fle_rot)

        # test rot90 close
        self.assertTrue(np.allclose(ims_cart_rot[0], ims_fle_rot[0], atol=1e-4))

        # 2Pi identity in FLE space (rotate by 2Pi)
        coeffs_fle_2pi = basis.rotate(coeffs, 2 * np.pi)
        ims_fle_2pi = basis.evaluate(coeffs_fle_2pi)

        # test 2Pi identity
        self.assertTrue(
            np.allclose(ims[0], ims_fle_2pi[0], atol=utest_tolerance(basis.dtype))
        )

        # Reflect in FLE space (rotate by Pi)
        coeffs_fle_pi = basis.rotate(coeffs, np.pi)
        ims_fle_pi = basis.evaluate(coeffs_fle_pi)

        # test reflection
        self.assertTrue(np.allclose(np.flipud(ims[0]), ims_fle_pi[0], atol=1e-4))

        # make sure you can pass in a 1-D array if you want
        _ = basis.lowpass(np.zeros((basis.count,)), np.pi)

        # cannot pass in the wrong number of coefficients
        with self.assertRaisesRegex(
            AssertionError, "Number of coefficients must match self.count."
        ):
            _ = basis.rotate(np.zeros((1, 10)), np.pi)

        # cannot pass in wrong shape
        with self.assertRaisesRegex(
            AssertionError,
            "Input a stack of coefficients of dimension",
        ):
            _ = basis.lowpass(np.zeros((3, 3, 3)), np.pi)

    def testRadialConvolution(self):
        # test ability to accurately convolve with a radial
        # (e.g. CTF) function via FLE coefficients

        L = 32
        basis = FLEBasis2D(L)
        # load test CTF
        ctf = np.load(os.path.join(DATA_DIR, "ctf_32x32.npy"))
        ctf = ctf / np.max(np.abs(ctf.flatten()))

        # get sample images
        ims = self.create_images(L, 10)
        # convolve using coefficients
        coeffs = basis.evaluate_t(ims)
        coeffs_convolved = basis.radialconv(coeffs, ctf)
        imgs_convolved_fle = basis.evaluate(coeffs_convolved).asnumpy()

        # convolve using FFT
        ctf = basis.evaluate(basis.evaluate_t(ctf)).asnumpy()
        ims = basis.evaluate(coeffs).asnumpy()

        imgs_convolved_slow = np.zeros((10, L, L))
        for i in range(10):
            ctf_pad = np.zeros((2 * L, 2 * L))
            ims_pad = np.zeros((2 * L, 2 * L))
            ctf_pad[L // 2 : L // 2 + L, L // 2 : L // 2 + L] = ctf[0, :, :]
            ims_pad[L // 2 : L // 2 + L, L // 2 : L // 2 + L] = ims[i, :, :]

            ctf_shift = fft.fftshift(ctf_pad.reshape(2 * L, 2 * L))
            ims_shift = fft.fftshift(ims_pad.reshape(2 * L, 2 * L))

            convolution_fft_pad = fft.fftshift(
                fft.ifft2(np.fft.fft2(ctf_shift) * np.fft.fft2(ims_shift))
            )
            imgs_convolved_slow[i, :, :] = np.real(
                convolution_fft_pad[L // 2 : L // 2 + L, L // 2 : L // 2 + L]
            )

        self.assertTrue(np.allclose(imgs_convolved_fle, imgs_convolved_slow, atol=1e-5))

    def create_images(self, L, n):
        # create sample data
        v = Volume(
            np.load(os.path.join(DATA_DIR, "clean70SRibosome_vol.npy")).astype(
                np.float64
            )
        )
        v = v.downsample(L)
        sim = Simulation(L=L, n=n, vols=v, dtype=v.dtype, seed=1103)
        img = sim.clean_images[:]
        return img

    def relerr(self, x, y):
        # relative error of two arrays
        x = np.array(x).flatten()
        y = np.array(y).flatten()
        return np.linalg.norm(x - y) / np.linalg.norm(x)
