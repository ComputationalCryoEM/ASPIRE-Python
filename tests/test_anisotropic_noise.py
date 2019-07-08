import numpy as np
from unittest import TestCase

from aspyre.source import SourceFilter
from aspyre.source.simulation import Simulation
from aspyre.imaging.filters import RadialCTFFilter
from aspyre.estimation.noise import AnisotropicNoiseEstimator

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

    def testAnisotropicNoisePSD(self):
        noise_estimator = AnisotropicNoiseEstimator(self.sim, batchSize=512)
        noise_psd = noise_estimator.estimate_noise_psd()
        self.assertTrue(np.allclose(
            noise_psd,
            [
                [+0.00112473, +0.00106200, +0.00118618, +0.00495772, +0.00495797, +0.00495772, +0.00118618, +0.00106200],
                [+0.00099063, +0.00113591, +0.00160692, +0.00462546, +0.00621764, +0.00475203, +0.00153705, +0.00116489],
                [+0.00113134, +0.00148855, +0.00267187, +0.00505812, +0.01086790, +0.00520619, +0.00271356, +0.00157493],
                [+0.00485551, +0.00453407, +0.00499355, +0.00672553, +0.01090170, +0.00696211, +0.00501925, +0.00460892],
                [+0.00506158, +0.00629060, +0.01099897, +0.01099300, +0.04534847, +0.01099300, +0.01099897, +0.00629060],
                [+0.00485551, +0.00460892, +0.00501925, +0.00696211, +0.01090170, +0.00672553, +0.00499355, +0.00453407],
                [+0.00113134, +0.00157493, +0.00271356, +0.00520619, +0.01086790, +0.00505812, +0.00267187, +0.00148855],
                [+0.00099063, +0.00116489, +0.00153705, +0.00475203, +0.00621764, +0.00462546, +0.00160692, +0.00113591],
            ]
        ))

    def testAnisotropicNoiseVariance(self):
        noise_estimator = AnisotropicNoiseEstimator(self.sim, batchSize=512)
        noise_variance = noise_estimator.estimate()
        self.assertAlmostEqual(0.04534846544265747, noise_variance)