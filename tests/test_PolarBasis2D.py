import numpy as np
from unittest import TestCase

from aspire.basis.polar_2d import PolarBasis2D

import os.path
DATA_DIR = os.path.join(os.path.dirname(__file__), 'saved_test_data')


class PolarBasis2DTestCase(TestCase):
    def setUp(self):
        self.basis = PolarBasis2D((8, 8), 4, 32)

    def tearDown(self):
        pass

    def testPolarBasis2DEvaluate_t(self):
        x = np.load(os.path.join(DATA_DIR, 'fbbasis_coefficients_8_8.npy'))
        pf = self.basis.evaluate_t(x)
        result = np.load(os.path.join(DATA_DIR, 'pfbasis_coefficients_8_4_32.npy'))
        self.assertTrue(np.allclose(pf, result))
