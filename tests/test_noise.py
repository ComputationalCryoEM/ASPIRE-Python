import logging
import os.path
from unittest import TestCase

import numpy as np
from parameterized import parameterized, parameterized_class

from aspire.image import CustomNoiseAdder, WhiteNoiseAdder
from aspire.noise import WhiteNoiseEstimator
from aspire.operators import FunctionFilter, RadialCTFFilter, ScaledFilter
from aspire.source.simulation import Simulation
from aspire.volume import AsymmetricVolume

DATA_DIR = os.path.join(os.path.dirname(__file__), "saved_test_data")

logger = logging.getLogger(__name__)


@parameterized_class(("L"), [(64,), (65,)])
class SimTestCase(TestCase):
    # Note L needs to be large enough that we have sufficient image "corners".
    L = 64

    def setUp(self):

        # Setup a sim with no noise, no ctf, no shifts,
        #   using a compactly supported volume.
        # ie, clean centered projections.
        self.sim = Simulation(
            vols=AsymmetricVolume(L=self.L, C=1).generate(),
            n=16,
            offsets=0,
        )

    def tearDown(self):
        pass

    def testWhiteNoise(self):
        noise_estimator = WhiteNoiseEstimator(self.sim, batchSize=512)
        noise_variance = noise_estimator.estimate()
        # Using a compactly supported volume should yield
        #   virtually no noise in the image corners.
        self.assertTrue(np.isclose(noise_variance, 0))


@parameterized_class(("L"), [(64,), (65,)])
class NoiseAdder(TestCase):
    L = 64
    dtype = np.float32

    def setUp(self):

        # Setup a sim with no noise, no ctf, no shifts,
        #   using a compactly supported volume.
        # ie, clean centered projections.
        self.sim = Simulation(
            vols=AsymmetricVolume(L=self.L, C=1, dtype=self.dtype).generate(),
            n=16,
            offsets=0,
        )

    def tearDown(self):
        pass

    @parameterized.expand([(10 ** (-x),) for x in range(1, 4)])
    def testCustomNoiseAdder(self, noise_var):
        """
        Custom Noise adder uses custom `Filter`.
        """
        logger.debug(
            f"testCustomNoiseAdder dtype={self.dtype} L={self.L} noise_var={noise_var}"
        )

        # def noise_function(x, y):
        #     return 1e-7 * np.exp(-(x * x + y * y) / (2 * 0.3**2))
        def brownish_spectrum(x, y):
            return np.minimum(1, 2 / (x * x + y * y + 1))

        brownish_filter = FunctionFilter(f=brownish_spectrum)
        # Scale the filter function to a target variance
        print("xxx", np.mean(brownish_filter.evaluate_grid(self.L)))
        m = np.mean(brownish_filter.evaluate_grid(self.L))
        scalar = noise_var / m
        custom_filter = ScaledFilter(brownish_filter, scalar)

        # Check we are achieving an estimate near the target
        self.sim.noise_adder = CustomNoiseAdder(noise_filter=custom_filter)
        est_noise_var = self.sim.noise_adder.noise_var
        logger.debug(f"Estimated Noise Variance {est_noise_var}")
        self.assertTrue(np.isclose(est_noise_var, noise_var, rtol=0.1))

        # noise_estimator = WhiteNoiseEstimator(self.sim, batchSize=512)
        # # Match estimate within 1%
        # self.assertTrue(np.isclose(noise_var, noise_estimator.estimate(), rtol=0.01))

    @parameterized.expand([(10 ** (-x),) for x in range(1, 4)])
    def testWhiteNoiseAdder(self, noise_var):
        logger.debug(
            f"testWhiteNoiseAdder dtype={self.dtype} L={self.L} noise_var={noise_var}"
        )
        self.sim.noise_adder = WhiteNoiseAdder(var=noise_var)

        # Assert we have passed through the var exactly
        self.assertTrue(self.sim.noise_adder.noise_var == noise_var)

        noise_estimator = WhiteNoiseEstimator(self.sim, batchSize=512)
        # Match estimate within 1%
        self.assertTrue(np.isclose(noise_var, noise_estimator.estimate(), rtol=0.01))
