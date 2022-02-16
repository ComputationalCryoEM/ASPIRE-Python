import os.path
from unittest import TestCase
from unittest.case import SkipTest

import numpy as np
from pytest import raises

from aspire.basis import FBBasis2D
from aspire.image import Image
from aspire.utils import complex_type, gaussian_2d, real_type, utest_tolerance
from aspire.utils.coor_trans import grid_2d
from aspire.utils.random import randn

DATA_DIR = os.path.join(os.path.dirname(__file__), "saved_test_data")


class FBBasis2DTestCase(TestCase):
    def setUp(self):
        self.dtype = np.float32
        self.L = 8
        self.basis = FBBasis2D((self.L, self.L), dtype=self.dtype)
        self.seed = 9161341

    def tearDown(self):
        pass

    def testFBBasis2DIndices(self):
        indices = self.basis.indices()

        self.assertTrue(
            np.allclose(
                indices["ells"],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    3.0,
                    3.0,
                    3.0,
                    3.0,
                    4.0,
                    4.0,
                    4.0,
                    4.0,
                    5.0,
                    5.0,
                    5.0,
                    5.0,
                    6.0,
                    6.0,
                    7.0,
                    7.0,
                    8.0,
                    8.0,
                ],
            )
        )

        self.assertTrue(
            np.allclose(
                indices["ks"],
                [
                    0.0,
                    1.0,
                    2.0,
                    3.0,
                    0.0,
                    1.0,
                    2.0,
                    0.0,
                    1.0,
                    2.0,
                    0.0,
                    1.0,
                    2.0,
                    0.0,
                    1.0,
                    2.0,
                    0.0,
                    1.0,
                    0.0,
                    1.0,
                    0.0,
                    1.0,
                    0.0,
                    1.0,
                    0.0,
                    1.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
            )
        )

        self.assertTrue(
            np.allclose(
                indices["sgns"],
                [
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    1.0,
                    1.0,
                    1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    1.0,
                    1.0,
                    -1.0,
                    -1.0,
                    1.0,
                    1.0,
                    -1.0,
                    -1.0,
                    1.0,
                    1.0,
                    -1.0,
                    -1.0,
                    1.0,
                    -1.0,
                    1.0,
                    -1.0,
                    1.0,
                    -1.0,
                ],
            )
        )

    def testFBBasis2DNorms(self):
        radial_norms, angular_norms = self.basis.norms()
        self.assertTrue(
            np.allclose(
                radial_norms * angular_norms,
                [
                    3.68065992303471,
                    2.41241466684800,
                    1.92454669738088,
                    1.64809729313301,
                    2.01913617828263,
                    1.50455726188833,
                    1.25183461029289,
                    1.70284654929000,
                    1.36051054373844,
                    1.16529703804363,
                    1.49532071137207,
                    1.25039038364830,
                    1.34537533748304,
                    1.16245357319190,
                    1.23042467443861,
                    1.09002083501080,
                    1.13867113286781,
                    1.06324777330476,
                    0.999841586390824,
                ],
            )
        )

    def testFBBasis2DEvaluate(self):
        coeffs = np.array(
            [
                1.07338590e-01,
                1.23690941e-01,
                6.44482039e-03,
                -5.40484306e-02,
                -4.85304586e-02,
                1.09852144e-02,
                3.87838396e-02,
                3.43796455e-02,
                -6.43284705e-03,
                -2.86677145e-02,
                -1.42313328e-02,
                -2.25684091e-03,
                -3.31840727e-02,
                -2.59706174e-03,
                -5.91919887e-04,
                -9.97433028e-03,
                9.19123928e-04,
                1.19891589e-03,
                7.49154982e-03,
                6.18865229e-03,
                -8.13265715e-04,
                -1.30715655e-02,
                -1.44160603e-02,
                2.90379956e-03,
                2.37066082e-02,
                4.88805735e-03,
                1.47870707e-03,
                7.63376018e-03,
                -5.60619559e-03,
                1.05165081e-02,
                3.30510143e-03,
                -3.48652120e-03,
                -4.23228797e-04,
                1.40484061e-02,
            ],
            dtype=self.dtype,
        )
        result = self.basis.evaluate(coeffs)

        self.assertTrue(
            np.allclose(
                result,
                np.load(
                    os.path.join(DATA_DIR, "fbbasis_evaluation_8_8.npy")
                ).T,  # RCOPT
                atol=utest_tolerance(self.dtype),
            )
        )

    def testFBBasis2DEvaluate_t(self):
        v = np.load(os.path.join(DATA_DIR, "fbbasis_coefficients_8_8.npy")).T  # RCOPT
        # While FB can accept arrays, prefable to pass FB2D and FFB2D Image instances.
        img = Image(v.astype(self.dtype))
        result = self.basis.evaluate_t(img)
        self.assertTrue(
            np.allclose(
                result,
                [
                    0.10761825,
                    0.12291151,
                    0.00836345,
                    -0.0619454,
                    -0.0483326,
                    0.01053718,
                    0.03977641,
                    0.03420101,
                    -0.0060131,
                    -0.02970658,
                    -0.0151334,
                    -0.00017575,
                    -0.03987446,
                    -0.00257069,
                    -0.0006621,
                    -0.00975174,
                    0.00108047,
                    0.00072022,
                    0.00753342,
                    0.00604493,
                    0.00024362,
                    -0.01711248,
                    -0.01387371,
                    0.00112805,
                    0.02407385,
                    0.00376325,
                    0.00081128,
                    0.00951368,
                    -0.00557536,
                    0.01087579,
                    0.00255393,
                    -0.00525156,
                    -0.00839695,
                    0.00802198,
                ],
                atol=utest_tolerance(self.dtype),
            )
        )

    def testFBBasis2DExpand(self):
        v = np.load(os.path.join(DATA_DIR, "fbbasis_coefficients_8_8.npy")).T  # RCOPT
        result = self.basis.expand(v.astype(self.dtype))
        self.assertTrue(
            np.allclose(
                result,
                [
                    0.10733859,
                    0.12369094,
                    0.00644482,
                    -0.05404843,
                    -0.04853046,
                    0.01098521,
                    0.03878384,
                    0.03437965,
                    -0.00643285,
                    -0.02866771,
                    -0.01423133,
                    -0.00225684,
                    -0.03318407,
                    -0.00259706,
                    -0.00059192,
                    -0.00997433,
                    0.00091912,
                    0.00119892,
                    0.00749155,
                    0.00618865,
                    -0.00081327,
                    -0.01307157,
                    -0.01441606,
                    0.00290380,
                    0.02370661,
                    0.00488806,
                    0.00147871,
                    0.00763376,
                    -0.00560620,
                    0.01051651,
                    0.00330510,
                    -0.00348652,
                    -0.00042323,
                    0.01404841,
                ],
                atol=utest_tolerance(self.dtype),
            )
        )

    def testGaussianExpand(self):
        # Offset slightly
        x0 = 0.50
        y0 = 0.75

        # Want sigma to be as large as possible without the Gaussian
        # spilling too much outside the central disk.
        sigma = self.L / 8
        im1 = gaussian_2d(self.L, x0=x0, y0=y0, sigma_x=sigma, sigma_y=sigma)
        im1 = im1.astype(self.dtype)

        coef = self.basis.expand(im1)
        im2 = self.basis.evaluate(coef)

        # For small L there's too much clipping at high freqs to get 1e-3
        # accuracy.
        if self.L < 32:
            atol = 1e-2
        else:
            atol = 1e-3

        self.assertTrue(np.allclose(im1, im2, atol=atol))

    def testIsotropic(self):
        sigma = self.L / 8
        im = gaussian_2d(self.L, sigma_x=sigma, sigma_y=sigma)
        im = im.astype(self.dtype)

        coef = self.basis.expand(im)

        ells = self.basis.indices()["ells"]

        energy_outside = np.sum(np.abs(coef[ells != 0]) ** 2)
        energy_total = np.sum(np.abs(coef) ** 2)

        energy_ratio = energy_outside / energy_total

        self.assertTrue(energy_ratio < 0.01)

    def testModulated(self):
        if self.L < 32:
            raise SkipTest

        ell = 1

        sigma = self.L / 8
        im = gaussian_2d(self.L, sigma_x=sigma, sigma_y=sigma)
        im = im.astype(self.dtype)

        g2d = grid_2d(self.L)

        for trig_fun in (np.sin, np.cos):
            im1 = im * trig_fun(ell * g2d["phi"])

            coef = self.basis.expand(im1)

            ells = self.basis.indices()["ells"]

            energy_outside = np.sum(np.abs(coef[ells != ell]) ** 2)
            energy_total = np.sum(np.abs(coef) ** 2)

            energy_ratio = energy_outside / energy_total

            self.assertTrue(energy_ratio < 0.10)

    def testEvaluateExpand(self):
        coef1 = randn(self.basis.count, seed=self.seed)
        coef1 = coef1.astype(self.dtype)

        im = self.basis.evaluate(coef1)
        coef2 = self.basis.expand(im)[:, 0]

        self.assertTrue(np.allclose(coef1, coef2, atol=utest_tolerance(self.dtype)))

    def testAdjoint(self):
        u = randn(self.basis.count, seed=self.seed)
        u = u.astype(self.dtype)

        Au = self.basis.evaluate(u)

        x = randn(*self.basis.sz, seed=self.seed)
        x = x.astype(self.dtype)

        ATx = self.basis.evaluate_t(x)

        Au_dot_x = np.sum(Au * x)
        u_dot_ATx = np.sum(u * ATx)

        self.assertTrue(np.isclose(Au_dot_x, u_dot_ATx))

    def testComplexCoversion(self):
        # Load a reasonable input
        x = np.load(os.path.join(DATA_DIR, "fbbasis_coefficients_8_8.npy"))

        # Express in an FB basis
        v1 = self.basis.expand(x.astype(self.dtype))

        # Convert real FB coef to complex coef,
        cv = self.basis.to_complex(v1)
        # then convert back to real coef representation.
        v2 = self.basis.to_real(cv)

        # The round trip should be equivalent up to machine precision
        self.assertTrue(np.allclose(v1, v2))

    def testComplexCoversionErrorsToComplex(self):
        # Load a reasonable input
        x = np.load(os.path.join(DATA_DIR, "fbbasis_coefficients_8_8.npy"))

        # Express in an FB basis
        v1 = self.basis.expand(x.astype(self.dtype))

        # Test catching Errors
        with raises(TypeError):
            # Pass complex into `to_complex`
            _ = self.basis.to_complex(v1.astype(np.complex64))

        # Test casting case, where basis and coef don't match
        if self.basis.dtype == np.float32:
            test_dtype = np.float64
        elif self.basis.dtype == np.float64:
            test_dtype = np.float32
        # Result should be same precision as coef input, just complex.
        result_dtype = complex_type(test_dtype)

        v3 = self.basis.to_complex(v1.astype(test_dtype))
        self.assertTrue(v3.dtype == result_dtype)

        # Try 0d vector, should not crash.
        _ = self.basis.to_complex(v1.reshape(-1))

    def testComplexCoversionErrorsToReal(self):
        # Load a reasonable input
        x = np.load(os.path.join(DATA_DIR, "fbbasis_coefficients_8_8.npy"))

        # Express in an FB basis
        cv1 = self.basis.to_complex(self.basis.expand(x.astype(self.dtype)))

        # Test catching Errors
        with raises(TypeError):
            # Pass real into `to_real`
            _ = self.basis.to_real(cv1.real.astype(np.float32))

        # Test casting case, where basis and coef precision don't match
        if self.basis.dtype == np.float32:
            test_dtype = np.complex128
        elif self.basis.dtype == np.float64:
            test_dtype = np.complex64
        # Result should be same precision as coef input, just real.
        result_dtype = real_type(test_dtype)

        v3 = self.basis.to_real(cv1.astype(test_dtype))
        self.assertTrue(v3.dtype == result_dtype)

        # Try a 0d vector, should not crash.
        _ = self.basis.to_real(cv1.reshape(-1))

    def testInitWithIntSize(self):
        # make sure we can instantiate with just an int as a shortcut
        self.assertEqual((8, 8), FBBasis2D(8).sz)
