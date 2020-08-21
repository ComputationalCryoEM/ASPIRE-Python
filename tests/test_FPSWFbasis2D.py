import os.path
from unittest import TestCase

import numpy as np

from aspire.basis.fpswf_2d import FPSWFBasis2D

DATA_DIR = os.path.join(os.path.dirname(__file__), 'saved_test_data')


class FPSWFBasis2DTestCase(TestCase):
    def setUp(self):
        self.basis = FPSWFBasis2D((8, 8), 1.0, 1.0)

    def tearDown(self):
        pass

    def testFPSWFBasis2DEvaluate_t(self):
        images = np.load(os.path.join(DATA_DIR, 'ffbbasis2d_xcoeff_in_8_8.npy')).T #RCOPT
        result = self.basis.evaluate_t(images)
        coeffs = np.load(os.path.join(DATA_DIR, 'fpswf2d_vcoeffs_out_8_8.npy'))
        # make sure both real and imaginary parts are consistent.
        self.assertTrue(np.allclose(np.real(result), np.real(coeffs)) and
                        np.allclose(np.imag(result)*1j, np.imag(coeffs)*1j))

    def testFPSWFBasis2DEvaluate(self):
        coeffs = np.load(os.path.join(DATA_DIR, 'fpswf2d_vcoeffs_out_8_8.npy'))
        result = self.basis.evaluate(coeffs)
        images = np.load(os.path.join(DATA_DIR, 'fpswf2d_xcoeffs_out_8_8.npy'))
        self.assertTrue(np.allclose(result, images))
