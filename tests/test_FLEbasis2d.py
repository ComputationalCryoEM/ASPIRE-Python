import os
from unittest import TestCase

import numpy as np
from scipy.io import loadmat, savemat

from aspire.basis import FLEBasis2D

from ._basis_util import UniversalBasisMixin

DATA_DIR = os.path.join(os.path.dirname(__file__), "saved_test_data")


class FLEBasis2DTestCase(TestCase, UniversalBasisMixin):
    L = 8
    dtype = np.float32

    def setUp(self):
        self.basis = FLEBasis2D((self.L, self.L), dtype=self.dtype)

    def testFastVDense(self):
        sz = 32
        basis = FLEBasis2D(32)
        dense_b = basis.create_dense_matrix()

        # load test data
        data_file = os.path.join(DATA_DIR, "fle_sample_32.mat")
        data = loadmat(data_file)
        x = data["x"]
        x = x / np.max(np.abs(x.flatten()))
        xvec = x.reshape((sz**2, 1))

        result_dense = dense_b.T @ xvec
        result_fast = basis.evaluate_t(x)
        np.save("fle_result.npy", result_fast)

        self.assertTrue(self.relerr(result_dense, result_fast) < 1e-8)

    def relerr(self, x, y):
        x = np.array(x).flatten()
        y = np.array(y).flatten()
        return np.linalg.norm(x - y) / np.linalg.norm(x)
