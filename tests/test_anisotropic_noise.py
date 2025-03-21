import os.path
from unittest import TestCase

import numpy as np

from aspire.noise import AnisotropicNoiseEstimator, WhiteNoiseEstimator
from aspire.operators import RadialCTFFilter
from aspire.source import ArrayImageSource, _LegacySimulation
from aspire.utils import utest_tolerance
from aspire.volume import LegacyVolume

DATA_DIR = os.path.join(os.path.dirname(__file__), "saved_test_data")


class SimTestCase(TestCase):
    def setUp(self):
        self.dtype = np.float32
        self.vol = LegacyVolume(L=8, dtype=self.dtype).generate()
        self.sim = _LegacySimulation(
            n=1024,
            vols=self.vol,
            unique_filters=[
                # Set legacy pixel size
                RadialCTFFilter(pixel_size=10, defocus=d)
                for d in np.linspace(1.5e4, 2.5e4, 7)
            ],
            dtype=self.dtype,
        )

        # Keep hardcoded tests passing after fixing swapped offsets.
        # See github issue #1146.
        self.sim.sim_offsets = self.sim.offsets[:, [1, 0]]
        self.sim = self.sim.update(offsets=self.sim.offsets[:, [1, 0]])

    def tearDown(self):
        pass

    def testAnisotropicNoisePSD(self):
        noise_estimator = AnisotropicNoiseEstimator(self.sim, batch_size=512)
        noise_psd = noise_estimator._estimate_noise_psd()
        self.assertTrue(
            np.allclose(
                noise_psd,
                [
                    [
                        +0.00112473,
                        +0.00106200,
                        +0.00118618,
                        +0.00495772,
                        +0.00495797,
                        +0.00495772,
                        +0.00118618,
                        +0.00106200,
                    ],
                    [
                        +0.00099063,
                        +0.00113591,
                        +0.00160692,
                        +0.00462546,
                        +0.00621764,
                        +0.00475203,
                        +0.00153705,
                        +0.00116489,
                    ],
                    [
                        +0.00113134,
                        +0.00148855,
                        +0.00267187,
                        +0.00505812,
                        +0.01086790,
                        +0.00520619,
                        +0.00271356,
                        +0.00157493,
                    ],
                    [
                        +0.00485551,
                        +0.00453407,
                        +0.00499355,
                        +0.00672553,
                        +0.01090170,
                        +0.00696211,
                        +0.00501925,
                        +0.00460892,
                    ],
                    [
                        +0.00506158,
                        +0.00629060,
                        +0.01099897,
                        +0.01099300,
                        +0.04534847,
                        +0.01099300,
                        +0.01099897,
                        +0.00629060,
                    ],
                    [
                        +0.00485551,
                        +0.00460892,
                        +0.00501925,
                        +0.00696211,
                        +0.01090170,
                        +0.00672553,
                        +0.00499355,
                        +0.00453407,
                    ],
                    [
                        +0.00113134,
                        +0.00157493,
                        +0.00271356,
                        +0.00520619,
                        +0.01086790,
                        +0.00505812,
                        +0.00267187,
                        +0.00148855,
                    ],
                    [
                        +0.00099063,
                        +0.00116489,
                        +0.00153705,
                        +0.00475203,
                        +0.00621764,
                        +0.00462546,
                        +0.00160692,
                        +0.00113591,
                    ],
                ],
                atol=utest_tolerance(self.sim.dtype),
            )
        )

    def testAnisotropicNoiseVariance(self):
        noise_estimator = AnisotropicNoiseEstimator(self.sim, batch_size=512)
        noise_variance = noise_estimator.estimate()
        self.assertTrue(
            np.allclose(
                0.005158715099241817,
                noise_variance,
                atol=utest_tolerance(self.sim.dtype),
            )
        )

    def testParseval(self):
        """
        Here we construct a source of white noise.
        Then code tests that the average noise power in the real domain,
        is equivalent to the sum of the magnitudes squared
        of all frequency coefficients in the Fourier domain.

        These are calculated by WhiteNoiseEstimator and
        AnisotropicNoiseEstimator respectively.

        See Parseval/Plancherel's Theorem.
        """

        wht_noise = np.random.randn(1024, 128, 128).astype(self.dtype)
        src = ArrayImageSource(wht_noise)

        wht_noise_estimator = WhiteNoiseEstimator(src, batch_size=512)
        wht_noise_variance = wht_noise_estimator.estimate()
        noise_estimator = AnisotropicNoiseEstimator(src, batch_size=512)
        noise_variance = noise_estimator.estimate()

        self.assertTrue(np.allclose(noise_variance, wht_noise_variance))
