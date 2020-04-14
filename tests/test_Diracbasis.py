import numpy as np
from unittest import TestCase

from aspire.basis.dirac import DiracBasis
from aspire.utils.matlab_compat import m_flatten

import os.path
DATA_DIR = os.path.join(os.path.dirname(__file__), 'saved_test_data')


class DiracBasisTestCase(TestCase):
    def setUp(self):
        self.basis = DiracBasis((8, 8))

    def tearDown(self):
        pass

    def testDiracEvaluate(self):
        v = np.load(os.path.join(DATA_DIR, 'fbbasis_coefficients_8_8.npy'))
        coeffs = m_flatten(v)
        result = self.basis.evaluate(coeffs)
        self.assertTrue(np.allclose(result, v))

    def testDiracEvaluate_t(self):
        x = np.load(os.path.join(DATA_DIR, 'fbbasis_coefficients_8_8.npy'))
        result = self.basis.evaluate_t(x)
        self.assertTrue(np.allclose(
            result,
            m_flatten(x)
        ))
