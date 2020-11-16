import os.path
from unittest import TestCase

import numpy as np

from aspire.operators import (
    CTFFilter,
    FunctionFilter,
    IdentityFilter,
    PowerFilter,
    RadialCTFFilter,
    ScalarFilter,
    ScaledFilter,
    ZeroFilter,
)
from aspire.utils import utest_tolerance

DATA_DIR = os.path.join(os.path.dirname(__file__), "saved_test_data")


class SimTestCase(TestCase):
    def setUp(self):
        self.dtype = np.float32
        # A 2 x 256 ndarray of spatial frequencies
        self.omega = np.load(os.path.join(DATA_DIR, "omega_2_256.npy"))

    def tearDown(self):
        pass

    def testFunctionFilter(self):
        filt = FunctionFilter(lambda x, y: np.exp(-(x ** 2 + y ** 2) / 2))
        result = filt.evaluate(self.omega)
        self.assertEqual(result.shape, (256,))
        self.assertTrue(
            np.allclose(
                result[:5],
                [
                    5.17231862e-05,
                    1.64432545e-04,
                    4.48039823e-04,
                    1.04633750e-03,
                    2.09436945e-03,
                ],
            )
        )

    def testZeroFilter(self):
        result = ZeroFilter().evaluate(self.omega)
        # For all filters, we should get a 1d ndarray back on evaluate
        self.assertEqual(result.shape, (256,))
        self.assertTrue(np.allclose(result, np.zeros(256)))

    def testIdentityFilter(self):
        result = IdentityFilter().evaluate(self.omega)
        # For all filters, we should get a 1d ndarray back on evaluate
        self.assertEqual(result.shape, (256,))
        self.assertTrue(np.allclose(result, np.ones(256)))

    def testScalarFilter(self):
        result = ScalarFilter(value=1.5).evaluate(self.omega)
        self.assertEqual(result.shape, (256,))
        self.assertTrue(np.allclose(result, np.repeat(1.5, 256)))

    def testPowerFilter(self):
        filt = PowerFilter(
            filter=FunctionFilter(lambda x, y: np.exp(-(x ** 2 + y ** 2) / 2)),
            power=0.5,
        )
        result = filt.evaluate(self.omega)
        self.assertEqual(result.shape, (256,))
        self.assertTrue(
            np.allclose(
                result[:5],
                np.array(
                    [
                        5.17231862e-05,
                        1.64432545e-04,
                        4.48039823e-04,
                        1.04633750e-03,
                        2.09436945e-03,
                    ]
                )
                ** 0.5,
            )
        )

    def testCTFFilter(self):
        filter = CTFFilter(defocus_u=1.5e4, defocus_v=1.5e4)
        result = filter.evaluate(self.omega)
        self.assertEqual(result.shape, (256,))

    def testScaledFilter(self):
        filt1 = CTFFilter(defocus_u=1.5e4, defocus_v=1.5e4)
        scale_value = 2.5
        result1 = filt1.evaluate(self.omega)
        # ScaledFilter scales the pixel size which cancels out
        # a corresponding scaling in omega
        filt2 = ScaledFilter(filt1, scale_value)
        result2 = filt2.evaluate(self.omega * scale_value)
        self.assertTrue(np.allclose(result1, result2, atol=utest_tolerance(self.dtype)))

    def testCTFScale(self):
        filt = CTFFilter(defocus_u=1.5e4, defocus_v=1.5e4)
        result1 = filt.evaluate(self.omega)
        scale_value = 2.5
        filt = filt.scale(scale_value)
        # scaling a CTFFilter scales the pixel size which cancels out
        # a corresponding scaling in omega
        result2 = filt.evaluate(self.omega * scale_value)
        self.assertTrue(np.allclose(result1, result2, atol=utest_tolerance(self.dtype)))

    def testRadialCTFFilter(self):
        filter = RadialCTFFilter(defocus=2.5e4)
        result = filter.evaluate(self.omega)
        self.assertEqual(result.shape, (256,))

    def testRadialCTFFilterGrid(self):
        filter = RadialCTFFilter(defocus=2.5e4)
        result = filter.evaluate_grid(8, dtype=self.dtype)

        self.assertEqual(result.shape, (8, 8))
        self.assertTrue(
            np.allclose(
                result,
                np.array(
                    [
                        [
                            0.461755701877834,
                            -0.995184514498978,
                            0.063120922443392,
                            0.833250206225063,
                            0.961464660252150,
                            0.833250206225063,
                            0.063120922443392,
                            -0.995184514498978,
                        ],
                        [
                            -0.995184514498978,
                            0.626977423649552,
                            0.799934516166400,
                            0.004814348317439,
                            -0.298096205735759,
                            0.004814348317439,
                            0.799934516166400,
                            0.626977423649552,
                        ],
                        [
                            0.063120922443392,
                            0.799934516166400,
                            -0.573061561512667,
                            -0.999286510416273,
                            -0.963805291282899,
                            -0.999286510416273,
                            -0.573061561512667,
                            0.799934516166400,
                        ],
                        [
                            0.833250206225063,
                            0.004814348317439,
                            -0.999286510416273,
                            -0.633095739808868,
                            -0.368890743119366,
                            -0.633095739808868,
                            -0.999286510416273,
                            0.004814348317439,
                        ],
                        [
                            0.961464660252150,
                            -0.298096205735759,
                            -0.963805291282899,
                            -0.368890743119366,
                            -0.070000000000000,
                            -0.368890743119366,
                            -0.963805291282899,
                            -0.298096205735759,
                        ],
                        [
                            0.833250206225063,
                            0.004814348317439,
                            -0.999286510416273,
                            -0.633095739808868,
                            -0.368890743119366,
                            -0.633095739808868,
                            -0.999286510416273,
                            0.004814348317439,
                        ],
                        [
                            0.063120922443392,
                            0.799934516166400,
                            -0.573061561512667,
                            -0.999286510416273,
                            -0.963805291282899,
                            -0.999286510416273,
                            -0.573061561512667,
                            0.799934516166400,
                        ],
                        [
                            -0.995184514498978,
                            0.626977423649552,
                            0.799934516166400,
                            0.004814348317439,
                            -0.298096205735759,
                            0.004814348317439,
                            0.799934516166400,
                            0.626977423649552,
                        ],
                    ]
                ),
                atol=utest_tolerance(self.dtype),
            )
        )

    def testRadialCTFFilterMultiplierGrid(self):
        filter = RadialCTFFilter(defocus=2.5e4) * RadialCTFFilter(defocus=2.5e4)
        result = filter.evaluate_grid(8, dtype=self.dtype)

        self.assertEqual(result.shape, (8, 8))
        self.assertTrue(
            np.allclose(
                result,
                np.array(
                    [
                        [
                            0.461755701877834,
                            -0.995184514498978,
                            0.063120922443392,
                            0.833250206225063,
                            0.961464660252150,
                            0.833250206225063,
                            0.063120922443392,
                            -0.995184514498978,
                        ],
                        [
                            -0.995184514498978,
                            0.626977423649552,
                            0.799934516166400,
                            0.004814348317439,
                            -0.298096205735759,
                            0.004814348317439,
                            0.799934516166400,
                            0.626977423649552,
                        ],
                        [
                            0.063120922443392,
                            0.799934516166400,
                            -0.573061561512667,
                            -0.999286510416273,
                            -0.963805291282899,
                            -0.999286510416273,
                            -0.573061561512667,
                            0.799934516166400,
                        ],
                        [
                            0.833250206225063,
                            0.004814348317439,
                            -0.999286510416273,
                            -0.633095739808868,
                            -0.368890743119366,
                            -0.633095739808868,
                            -0.999286510416273,
                            0.004814348317439,
                        ],
                        [
                            0.961464660252150,
                            -0.298096205735759,
                            -0.963805291282899,
                            -0.368890743119366,
                            -0.070000000000000,
                            -0.368890743119366,
                            -0.963805291282899,
                            -0.298096205735759,
                        ],
                        [
                            0.833250206225063,
                            0.004814348317439,
                            -0.999286510416273,
                            -0.633095739808868,
                            -0.368890743119366,
                            -0.633095739808868,
                            -0.999286510416273,
                            0.004814348317439,
                        ],
                        [
                            0.063120922443392,
                            0.799934516166400,
                            -0.573061561512667,
                            -0.999286510416273,
                            -0.963805291282899,
                            -0.999286510416273,
                            -0.573061561512667,
                            0.799934516166400,
                        ],
                        [
                            -0.995184514498978,
                            0.626977423649552,
                            0.799934516166400,
                            0.004814348317439,
                            -0.298096205735759,
                            0.004814348317439,
                            0.799934516166400,
                            0.626977423649552,
                        ],
                    ]
                )
                ** 2,
                atol=utest_tolerance(self.dtype),
            )
        )

    def testDualFilter(self):
        ctf_filter = CTFFilter(defocus_u=1.5e4, defocus_v=1.5e4)
        result = ctf_filter.evaluate(-self.omega)
        dual_filter = ctf_filter.dual()
        dual_result = dual_filter.evaluate(self.omega)
        self.assertTrue(np.allclose(result, dual_result))
