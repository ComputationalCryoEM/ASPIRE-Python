import os.path
from unittest import TestCase

import numpy as np

from aspire.noise import WhiteNoiseEstimator
from aspire.operators import RadialCTFFilter
from aspire.source.simulation import Simulation
from aspire.volume import AsymmetricVolume

DATA_DIR = os.path.join(os.path.dirname(__file__), "saved_test_data")

class NoiseAdder(TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def CustomNoiseAdder(self):
        pass

    def WhiteNoiseAdder(self):
        pass
    

class SimTestCase(TestCase):
    def setUp(self):

        self.L = 32
        # Setup a sim with no noise, no ctf, no shifts,
        #   using a compactly supported volume.
        # These should be centered projections.
        self.sim = Simulation(
            vols=AsymmetricVolume(32,1).generate(),
            n=1024,
            offsets=0,
        )

    def tearDown(self):
        pass

    #@pytest.mark.skip(reason="I believe this code is broken")
    def testWhiteNoise(self):
        noise_estimator = WhiteNoiseEstimator(self.sim, batchSize=512)
        noise_variance = noise_estimator.estimate()
        self.assertAlmostEqual(0, self.noise_var)

    
