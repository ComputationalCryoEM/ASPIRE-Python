import os.path
from unittest import TestCase

import numpy as np

from aspire.noise import WhiteNoiseEstimator
from aspire.operators import RadialCTFFilter
from aspire.source.simulation import Simulation

DATA_DIR = os.path.join(os.path.dirname(__file__), "saved_test_data")


class SimTestCase(TestCase):
    def setUp(self):
        self.sim = Simulation(
            n=1024,
            unique_filters=[
                RadialCTFFilter(defocus=d) for d in np.linspace(1.5e4, 2.5e4, 7)
            ],
        )

    def tearDown(self):
        pass

    def testWhiteNoise(self):
        noise_estimator = WhiteNoiseEstimator(self.sim, batchSize=512)
        noise_variance = noise_estimator.estimate()
        self.assertAlmostEqual(noise_variance, 0.00307627)
