import logging
import os
from unittest import TestCase

import numpy as np
import pytest

from aspire.denoising import adaptive_support
from aspire.image import Image
from aspire.utils import gaussian_2d

logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "saved_test_data")


class AdaptiveSupportTest(TestCase):
    def setUp(self):
        self.images = Image(
            np.load(os.path.join(DATA_DIR, "sim_images_with_noise.npy"))
        )

    def test_adaptivate_support(self):
        """
        The following low res references were computed after manually validating
        against a higher resolution example in MATLAB.
        """
        references = {0.8: (1 / 3, 2), 0.95: (1 / 3, 3), 0.99: (0.5, 3)}

        for threshold, reference in references.items():
            self.assertTrue(
                np.allclose(adaptive_support(self.images, threshold), reference)
            )

    def testAdaptiveSupportBadThreshold(self):
        """
        Method should raise meaningful error when passed unreasonable thresholds.
        """

        with pytest.raises(ValueError, match=r"Given energy_threshold.*"):
            _ = adaptive_support(self.images, -0.5)

        with pytest.raises(ValueError, match=r"Given energy_threshold.*"):
            _ = adaptive_support(self.images, 9000)

    def testAdaptiveSupportNonImageInput(self):
        """
        Method should raise meaningful error when passed wrong format input.
        """

        with pytest.raises(
            RuntimeError, match="adaptive_support expects Image instance"
        ):
            # Pass numpy array.
            _ = adaptive_support(self.images.asnumpy())

    def test_adaptivate_support_gaussian(self):
        """
        Test against known Gaussians.
        """

        resolution = 65
        sigma = 64
        n_disc = 10
        discs = np.tile(
            gaussian_2d(resolution, sigma_x=sigma, sigma_y=sigma), (n_disc, 1, 1)
        )
        disc_img = Image(discs)

        threshold = 1
        np.allclose(adaptive_support(disc_img, threshold), (0.5, 65 // 2))
