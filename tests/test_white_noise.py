import numpy as np
from unittest import TestCase

from aspyre.source import SourceFilter
from aspyre.source.simulation import Simulation
from aspyre.imaging.filters import RadialCTFFilter
from aspyre.estimation.noise import WhiteNoiseEstimator

import os.path
DATA_DIR = os.path.join(os.path.dirname(__file__), 'saved_test_data')


class SimTestCase(TestCase):
    def setUp(self):
        self.sim = Simulation(
            n=1024,
            filters=SourceFilter(
                [RadialCTFFilter(defocus=d) for d in np.linspace(1.5e4, 2.5e4, 7)],
                n=1024
            )
        )

    def tearDown(self):
        pass

    def testWhiteNoise(self):
        noise_estimator = WhiteNoiseEstimator(self.sim, batchSize=512)
        noise_variance = noise_estimator.estimate()
        self.assertAlmostEqual(noise_variance, 0.00307627)



