import os.path
from unittest import TestCase

import numpy as np

from aspire.utils.filters import (CTFFilter, FunctionFilter, IdentityFilter,
                                  PowerFilter, RadialCTFFilter, ScalarFilter,
                                  ZeroFilter)

DATA_DIR = os.path.join(os.path.dirname(__file__), 'saved_test_data')


class SimTestCase(TestCase):
    def setUp(self):
        self.dtype = np.float64
        # A 2 x 256 ndarray of spatial frequencies
        self.omega = np.load(os.path.join(DATA_DIR, 'omega_2_256.npy'))

    def tearDown(self):
        pass

    def testFunctionFilter(self):
        filt = FunctionFilter(lambda x, y: np.exp(-(x**2 + y**2) / 2))
        result = filt.evaluate(self.omega)
        self.assertEqual(result.shape, (256,))
        self.assertTrue(np.allclose(
            result[:5],
            [5.17231862e-05, 1.64432545e-04, 4.48039823e-04, 1.04633750e-03, 2.09436945e-03]
        ))

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
        filt = PowerFilter(filter=FunctionFilter(lambda x, y: np.exp(-(x**2 + y**2) / 2)), power=0.5)
        result = filt.evaluate(self.omega)
        self.assertEqual(result.shape, (256,))
        self.assertTrue(np.allclose(
            result[:5],
            np.array([5.17231862e-05, 1.64432545e-04, 4.48039823e-04, 1.04633750e-03, 2.09436945e-03])**0.5
        ))

    def testCTFFilter(self):
        filter = CTFFilter(defocus_u=1.5e4, defocus_v=1.5e4)
        result = filter.evaluate(self.omega)
        self.assertEqual(result.shape, (256,))

    def testRadialCTFFilter(self):
        filter = RadialCTFFilter(defocus=2.5e4)
        result = filter.evaluate(self.omega)
        self.assertEqual(result.shape, (256,))

    def testRadialCTFFilterGrid(self):
        filter = RadialCTFFilter(defocus=2.5e4)
        result = filter.evaluate_grid(8, dtype=self.dtype)

        self.assertEqual(result.shape, (8, 8))
        self.assertTrue(np.allclose(
            result,
            np.array([
                [ 0.461755701877834,  -0.995184514498978,   0.063120922443392,   0.833250206225063,   0.961464660252150,   0.833250206225063,   0.063120922443392,  -0.995184514498978],
                [-0.995184514498978,   0.626977423649552,   0.799934516166400,   0.004814348317439,  -0.298096205735759,   0.004814348317439,   0.799934516166400,   0.626977423649552],
                [ 0.063120922443392,   0.799934516166400,  -0.573061561512667,  -0.999286510416273,  -0.963805291282899,  -0.999286510416273,  -0.573061561512667,   0.799934516166400],
                [ 0.833250206225063,   0.004814348317439,  -0.999286510416273,  -0.633095739808868,  -0.368890743119366,  -0.633095739808868,  -0.999286510416273,   0.004814348317439],
                [ 0.961464660252150,  -0.298096205735759,  -0.963805291282899,  -0.368890743119366,  -0.070000000000000,  -0.368890743119366,  -0.963805291282899,  -0.298096205735759],
                [ 0.833250206225063,   0.004814348317439,  -0.999286510416273,  -0.633095739808868,  -0.368890743119366,  -0.633095739808868,  -0.999286510416273,   0.004814348317439],
                [ 0.063120922443392,   0.799934516166400,  -0.573061561512667,  -0.999286510416273,  -0.963805291282899,  -0.999286510416273,  -0.573061561512667,   0.799934516166400],
                [-0.995184514498978,   0.626977423649552,   0.799934516166400,   0.004814348317439,  -0.298096205735759,   0.004814348317439,   0.799934516166400,   0.626977423649552]
            ]),
            atol = 1e-6 if self.dtype == np.float32 else 1e-8
        ))

    def testRadialCTFFilterMultiplierGrid(self):
        filter = RadialCTFFilter(defocus=2.5e4) * RadialCTFFilter(defocus=2.5e4)
        result = filter.evaluate_grid(8, dtype=self.dtype)

        self.assertEqual(result.shape, (8, 8))
        self.assertTrue(np.allclose(
            result,
            np.array([
                [ 0.461755701877834,  -0.995184514498978,   0.063120922443392,   0.833250206225063,   0.961464660252150,   0.833250206225063,   0.063120922443392,  -0.995184514498978],
                [-0.995184514498978,   0.626977423649552,   0.799934516166400,   0.004814348317439,  -0.298096205735759,   0.004814348317439,   0.799934516166400,   0.626977423649552],
                [ 0.063120922443392,   0.799934516166400,  -0.573061561512667,  -0.999286510416273,  -0.963805291282899,  -0.999286510416273,  -0.573061561512667,   0.799934516166400],
                [ 0.833250206225063,   0.004814348317439,  -0.999286510416273,  -0.633095739808868,  -0.368890743119366,  -0.633095739808868,  -0.999286510416273,   0.004814348317439],
                [ 0.961464660252150,  -0.298096205735759,  -0.963805291282899,  -0.368890743119366,  -0.070000000000000,  -0.368890743119366,  -0.963805291282899,  -0.298096205735759],
                [ 0.833250206225063,   0.004814348317439,  -0.999286510416273,  -0.633095739808868,  -0.368890743119366,  -0.633095739808868,  -0.999286510416273,   0.004814348317439],
                [ 0.063120922443392,   0.799934516166400,  -0.573061561512667,  -0.999286510416273,  -0.963805291282899,  -0.999286510416273,  -0.573061561512667,   0.799934516166400],
                [-0.995184514498978,   0.626977423649552,   0.799934516166400,   0.004814348317439,  -0.298096205735759,   0.004814348317439,   0.799934516166400,   0.626977423649552]
            ])**2,
            atol = 1e-6 if self.dtype == np.float32 else 1e-8
        ))

    def testDualFilter(self):
        ctf_filter = CTFFilter(defocus_u=1.5e4, defocus_v=1.5e4)
        result = ctf_filter.evaluate(-self.omega)
        dual_filter = ctf_filter.dual()
        dual_result = dual_filter.evaluate(self.omega)
        self.assertTrue(np.allclose(result, dual_result))
