import numpy as np
from unittest import TestCase
import os.path

from aspire.basis.pswf_2d import PSWFBasis2D


DATA_DIR = os.path.join(os.path.dirname(__file__), 'saved_test_data')


class PSWFBasis2DTestCase(TestCase):
    def setUp(self):
        self.basis = PSWFBasis2D((8, 8), 1.0, 1.0)

    def tearDown(self):
        pass

    def testPSWFBasis2DEvaluate_t(self):
        images = np.load(os.path.join(DATA_DIR, 'ffbbasis2d_xcoeff_in_8_8.npy'))
        result = self.basis.evaluate_t(images)
        coeffs = np.load(os.path.join(DATA_DIR, 'pswf2d_vcoeffs_out_8_8.npy'))
        # make sure both real and imaginary parts are consistent.
        self.assertTrue(np.allclose(np.real(result), np.real(coeffs)) and
                        np.allclose(np.imag(result) * 1j, np.imag(coeffs) * 1j))

    def testPSWFBasis2DEvaluate(self):
        coeffs = np.load(os.path.join(DATA_DIR, 'pswf2d_vcoeffs_out_8_8.npy'))
        result = self.basis.evaluate(coeffs)
        images = np.load(os.path.join(DATA_DIR, 'pswf2d_xcoeff_out_8_8.npy'))
        self.assertTrue(np.allclose(result, images))
