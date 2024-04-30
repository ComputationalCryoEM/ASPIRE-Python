import logging
import os.path
from unittest import TestCase

import numpy as np

from aspire.basis import Coef, FBBasis3D
from aspire.operators import RadialCTFFilter
from aspire.reconstruction import WeightedVolumesEstimator
from aspire.source import Simulation
from aspire.utils import grid_3d

logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "saved_test_data")


class WeightedVolumesEstimatorTestCase(TestCase):
    def setUp(self):
        self.dtype = np.float32
        self.n = 512
        self.r = 2
        self.L = L = 8
        self.sim = Simulation(
            n=self.n,
            C=1,  # single volume
            unique_filters=[
                RadialCTFFilter(defocus=d) for d in np.linspace(1.5e4, 2.5e4, 7)
            ],
            dtype=self.dtype,
            seed=1617,
        )
        # Todo, swap for default FFB3D
        self.basis = FBBasis3D((L, L, L), dtype=self.dtype)
        self.weights = np.ones((self.n, self.r)) / np.sqrt(self.n)
        self.estimator = WeightedVolumesEstimator(
            self.weights, self.sim, basis=self.basis, preconditioner="none"
        )
        self.estimator_with_preconditioner = WeightedVolumesEstimator(
            self.weights, self.sim, basis=self.basis, preconditioner="circulant"
        )
        self.mask = grid_3d(self.L)["r"] < 1

    def tearDown(self):
        pass

    def testPositiveWeightedEstimates(self):
        estimate = self.estimator.estimate()

        est = estimate * self.mask
        vol = self.sim.vols.asnumpy() * self.mask
        vol /= np.linalg.norm(vol)

        # Compare each output volume
        for _est in est:
            np.testing.assert_allclose(
                _est / np.linalg.norm(_est), vol / np.linalg.norm(vol), atol=0.1
            )

    def testAdjoint(self):
        # Mean coefs formed by backprojections
        mean_b_coef = self.estimator.src_backward()

        # Evaluate mean coefs into a volume
        est = Coef(self.basis, mean_b_coef).evaluate()

        # Mask off corners of volume
        vol = self.sim.vols.asnumpy() * self.mask

        # Assert the mean volumes are close to original volume
        for _est in est:
            np.testing.assert_allclose(
                _est / np.linalg.norm(_est), vol / np.linalg.norm(vol), atol=0.1
            )

    def testOptimize1(self):
        """
        x = self.estimator.conj_grad(mean_b_coef)
        """

    def testOptimize2(self):
        """
        x = self.estimator_with_preconditioner.conj_grad(mean_b_coef)
        """

    def testNegativeWeightedEstimates(self):
        """
        Here we'll test createing two volumes.
        One with positive and another with negative weights.
        """
        weights = np.ones((self.n, self.r)) / np.sqrt(self.n)
        weights[:, 1] *= -1  # negate second set of weights

        estimator = WeightedVolumesEstimator(
            weights, self.sim, basis=self.basis, preconditioner="none"
        )

        estimate = estimator.estimate()

        est = estimate * self.mask
        vol = self.sim.vols.asnumpy() * self.mask
        vol /= np.linalg.norm(vol)

        # Compare positive weighted output volume
        np.testing.assert_allclose(
            est[0] / np.linalg.norm(est[0]), vol / np.linalg.norm(vol), atol=0.1
        )

        # Compare negative weighted output volume
        np.testing.assert_allclose(
            -1 * est[1] / np.linalg.norm(est[1]), vol / np.linalg.norm(vol), atol=0.1
        )
