import logging
import os
from unittest import TestCase

import numpy as np
import pytest

from aspire.denoising import adaptive_support
from aspire.image import Image
from aspire.source import ArrayImageSource
from aspire.utils import gaussian_2d

logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "saved_test_data")


class AdaptiveSupportTest(TestCase):
    def setUp(self):

        self.resolution = 1025
        self.sigma = 128
        n_disc = 10
        discs = np.tile(
            gaussian_2d(self.resolution, sigma_x=self.sigma, sigma_y=self.sigma),
            (n_disc, 1, 1),
        )
        self.img_src = ArrayImageSource(Image(discs))

    def testAdaptiveSupportBadThreshold(self):
        """
        Method should raise meaningful error when passed unreasonable thresholds.
        """

        with pytest.raises(ValueError, match=r"Given energy_threshold.*"):
            _ = adaptive_support(self.img_src, -0.5)

        with pytest.raises(ValueError, match=r"Given energy_threshold.*"):
            _ = adaptive_support(self.img_src, 9000)

    def testAdaptiveSupportIncorrectInput(self):
        """
        Method should raise meaningful error when passed wrong format input.
        """

        with pytest.raises(
            RuntimeError,
            match="adaptive_support expects `Source` instance or subclass.",
        ):
            # Pass numpy array.
            _ = adaptive_support(np.empty((10, 32, 32)))

    def test_adaptivate_support_gaussian(self):
        """
        Test against known Gaussians.
        """

        references = {
            1: 0.68,
            2: 0.96,
            3: 0.999,  # slightly off
            self.resolution / (2 * self.sigma): 1,
        }

        # for one, two, three, inf standard deviations (one sided)
        for stddevs, threshold in references.items():
            # Real support should be ~ 1, 2, 3 times sigma
            c, r = adaptive_support(self.img_src, threshold)
            # Check closer to this threshold than next
            logger.info(f"{c} {r} = adaptive_support(..., {threshold})")
            self.assertTrue(np.abs(r / self.sigma - stddevs) < 0.5)
