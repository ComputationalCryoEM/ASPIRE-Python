import os.path
from unittest import TestCase

import numpy as np

from aspire.basis.pswf_utils import BNMatrix

DATA_DIR = os.path.join(os.path.dirname(__file__), "saved_test_data")


class BNMatrixTestCase(TestCase):
    def setUp(self):
        big_n = 0
        bandlimit = 1.0 * np.pi * 8
        approx_length = 20

        self.bnmatrix = BNMatrix(big_n, bandlimit, approx_length)

    def tearDown(self):
        pass

    def testBNMatrixGetEigVectors(self):
        v, w = self.bnmatrix.get_eig_vectors()
        eig_vectors = np.load(os.path.join(DATA_DIR, "bnmatrix_eig_vectors.npy"))
        eig_values = np.load(os.path.join(DATA_DIR, "bnmatrix_eig_values.npy"))
        self.assertTrue(np.allclose(v, eig_vectors) and np.allclose(w, eig_values))

    def testBNMatrixDenseMat(self):
        result = self.bnmatrix.dense_mat()
        dense_mat = np.load(os.path.join(DATA_DIR, "bnmatrix_dense_mat.npy"))
        self.assertTrue(np.allclose(dense_mat, result))
