import os
from unittest import TestCase

import numpy as np
from parameterized import parameterized

from aspire.basis import FLEBasis2D
from aspire.image import Image
from aspire.source import Simulation
from aspire.volume import Volume

from ._basis_util import UniversalBasisMixin

DATA_DIR = os.path.join(os.path.dirname(__file__), "saved_test_data")


class FLEBasis2DTestCase(TestCase, UniversalBasisMixin):
    L = 8
    dtype = np.float32

    def setUp(self):
        self.basis = FLEBasis2D((self.L, self.L), dtype=self.dtype)

    # even and odd images with all guaranteed epsilons from paper
    @parameterized.expand(
        [
            [32, 1e-4],
            [32, 1e-7],
            [32, 1e-10],
            [32, 1e-14],
            [33, 1e-4],
            [33, 1e-7],
            [33, 1e-10],
            [33, 1e-14],
        ]
    )
    # check closeness guarantees for fast vs dense matrix method
    def testFastVDense(self, L, epsilon):
        basis = FLEBasis2D(L, epsilon=epsilon, dtype=np.float64)
        dense_b = basis.create_dense_matrix()

        # create sample particle
        x = self.create_image(L).asnumpy()
        x = x / np.max(np.abs(x.flatten()))
        xvec = x.reshape((L**2, 1))

        # explicit matrix multiplication
        result_dense = dense_b.T @ xvec
        # fast evaluate_t
        result_fast = basis.evaluate_t(Image(x))

        relerr = self.relerr(result_dense.T, result_fast)
        self.assertTrue(relerr < epsilon)

    def create_image(self, L):
        v = Volume(
            np.load(os.path.join(DATA_DIR, "clean70SRibosome_vol.npy")).astype(
                np.float64
            )
        )
        v = v.downsample(L)
        sim = Simulation(L=L, n=1, vols=v, dtype=v.dtype, seed=1103)
        img = sim.clean_images[0]
        return img

    def relerr(self, x, y):
        x = np.array(x).flatten()
        y = np.array(y).flatten()
        return np.linalg.norm(x - y) / np.linalg.norm(x)
