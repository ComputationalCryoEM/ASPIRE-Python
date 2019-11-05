import numpy as np
from unittest import TestCase

from aspire.basis.fpswf_2d import FPSWFBasis2D

import os.path
DATA_DIR = os.path.join(os.path.dirname(__file__), 'saved_test_data')

class FPSWFBasis2DTestCase(TestCase):
    def setUp(self):
        self.basis = FPSWFBasis2D((129, 129), 1.0, 1.0)

    def tearDown(self):
        pass

    def testFPSWFBasis2DEvaluate_t(self):
        images = np.load(os.path.join(DATA_DIR, 'example_data_np_array.npy'))
        result = self.basis.evaluate_t(images)
        coeffs = np.load(os.path.join(DATA_DIR, 'example_data_fpswf2d_coeffs64.npy'))
        # make sure both real and imaginary parts are consistent.
        self.assertTrue(np.allclose(np.real(result), np.real(coeffs)) and
                        np.allclose(np.imag(result)*1j, np.imag(coeffs)*1j))

    def testFPSWFBasis2DEvaluate(self):
        coeffs = np.load(os.path.join(DATA_DIR, 'example_data_fpswf2d_coeffs64.npy'))
        result = self.basis.evaluate(coeffs)
        images = np.load(os.path.join(DATA_DIR, 'example_data_fpswf2d_reconsImgs64.npy'))
        self.assertTrue(np.allclose(result, images))





