import logging
import os.path
from unittest import TestCase

import numpy as np
import pytest

from aspire.noise import AnisotropicNoiseEstimator
from aspire.operators.filters import FunctionFilter, RadialCTFFilter
from aspire.source import ArrayImageSource
from aspire.source.simulation import Simulation
from aspire.utils import grid_2d
from aspire.utils.matrix import anorm

DATA_DIR = os.path.join(os.path.dirname(__file__), "saved_test_data")


class PreprocessPLTestCase(TestCase):
    def setUp(self):

        self.L = 64
        self.n = 128
        self.dtype = np.float32
        self.noise_filter = FunctionFilter(lambda x, y: np.exp(-(x**2 + y**2) / 2))

        self.sim = Simulation(
            L=self.L,
            n=self.n,
            unique_filters=[
                RadialCTFFilter(defocus=d) for d in np.linspace(1.5e4, 2.5e4, 7)
            ],
            noise_filter=self.noise_filter,
            dtype=self.dtype,
        )
        self.imgs_org = self.sim.images[: self.n]

    # This is a workaround to use a `pytest` fixture with `unittest` style cases.
    # We use it below to capture and inspect the log
    @pytest.fixture(autouse=True)
    def inject_fixtures(self, caplog):
        self._caplog = caplog

    def testPhaseFlip(self):
        self.sim.phase_flip()
        imgs_pf = self.sim.images[: self.n]

        # check energy conservation
        self.assertTrue(
            np.allclose(
                anorm(self.imgs_org.asnumpy(), axes=(1, 2)),
                anorm(imgs_pf.asnumpy(), axes=(1, 2)),
            )
        )

    def testEmptyPhaseFlip(self):
        """
        Attempting phase_flip without CTFFilters should raise.
        """

        # Create a Simulation without any CTFFilters
        sim = Simulation(
            L=self.L,
            n=self.n,
            noise_filter=self.noise_filter,
            dtype=self.dtype,
        )

        # Test we warn
        with self._caplog.at_level(logging.WARN):
            sim.phase_flip()
            assert "No Filters found" in self._caplog.text

    def testNormBackground(self):
        bg_radius = 1.0
        grid = grid_2d(self.L, indexing="yx")
        mask = grid["r"] > bg_radius
        self.sim.normalize_background()
        imgs_nb = self.sim.images[: self.n].asnumpy()
        new_mean = np.mean(imgs_nb[:, mask])
        new_variance = np.var(imgs_nb[:, mask])

        # new mean of noise should be close to zero and variance should be close to 1
        self.assertTrue(new_mean < 1e-7 and abs(new_variance - 1) < 1e-7)

    def testWhiten(self):
        noise_estimator = AnisotropicNoiseEstimator(self.sim)
        self.sim.whiten(noise_estimator.filter)
        imgs_wt = self.sim.images[: self.n].asnumpy()

        # calculate correlation between two neighboring pixels from background
        corr_coef = np.corrcoef(
            imgs_wt[:, self.L - 1, self.L - 1], imgs_wt[:, self.L - 2, self.L - 1]
        )

        # correlation matrix should be close to identity
        self.assertTrue(np.allclose(np.eye(2), corr_coef, atol=1e-1))

    def testWhiten2(self):
        # Excercises missing cases using odd image resolutions with filter.
        #  Relates to GitHub issue #401.
        # Otherwise this is the same as testWhiten, though the accuracy
        #  (atol) for odd resolutions seems slightly worse.
        L = self.L - 1
        assert L % 2 == 1, "Test resolution should be odd"

        sim = Simulation(
            L=L,
            n=self.n,
            unique_filters=[
                RadialCTFFilter(defocus=d) for d in np.linspace(1.5e4, 2.5e4, 7)
            ],
            noise_filter=self.noise_filter,
            dtype=self.dtype,
        )
        noise_estimator = AnisotropicNoiseEstimator(sim)
        sim.whiten(noise_estimator.filter)
        imgs_wt = sim.images[: self.n].asnumpy()

        corr_coef = np.corrcoef(imgs_wt[:, L - 1, L - 1], imgs_wt[:, L - 2, L - 1])

        # Correlation matrix should be close to identity
        self.assertTrue(np.allclose(np.eye(2), corr_coef, atol=2e-1))

    def testInvertContrast(self):
        sim1 = self.sim
        imgs1 = sim1.images[:128]
        sim1.invert_contrast()
        imgs1_rc = sim1.images[:128]
        # need to set the negative images to the second simulation object
        sim2 = ArrayImageSource(-imgs1)
        sim2.invert_contrast()
        imgs2_rc = sim2.images[:128]

        # all images should be the same after inverting contrast
        self.assertTrue(np.allclose(imgs1_rc.asnumpy(), imgs2_rc.asnumpy()))
