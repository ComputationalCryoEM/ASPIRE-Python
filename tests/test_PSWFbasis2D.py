import numpy as np
from unittest import TestCase

from aspire.basis.pswf_2d import PSWFBasis2D

import os.path
DATA_DIR = os.path.join(os.path.dirname(__file__), 'saved_test_data')

class PSWFBasis2DTestCase(TestCase):
    def setUp(self):
        self.basis = PSWFBasis2D((129, 129), 1.0, 1.0)

    def tearDown(self):
        pass

    def testPSWFBasis2DEvaluate_t(self):
        images = np.load(os.path.join(DATA_DIR, 'example_data_np_array.npy'))
        result = self.basis.evaluate_t(images)
        coeffs = np.load(os.path.join(DATA_DIR, 'example_data_pswf2d_coeffs64.npy'))
        # make sure both real and imaginary parts are consistent.
        self.assertTrue(np.allclose(np.real(result), np.real(coeffs)) and
                        np.allclose(np.imag(result)*1j, np.imag(coeffs)*1j))

    def testPSWFBasis2DEvaluate(self):
        coeffs = np.load(os.path.join(DATA_DIR, 'example_data_pswf2d_coeffs64.npy'))
        result = self.basis.evaluate(coeffs)
        images = np.load(os.path.join(DATA_DIR, 'example_data_pswf2d_reconsImgs64.npy'))
        self.assertTrue(np.allclose(result, images))

    def testPSWFBasis2DDiff(self):
        images = np.load(os.path.join(DATA_DIR, 'example_data_np_array.npy'))
        coeffs = self.basis.evaluate_t(images)
        result = self.basis.evaluate(coeffs)
        maxdiff = np.max(abs(result[..., 0]-images[..., 0]))
        print(maxdiff)
        self.assertTrue(maxdiff < 0.01)




