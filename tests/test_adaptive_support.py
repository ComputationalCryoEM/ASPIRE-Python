import logging
import os
from unittest import TestCase

import numpy as np
import pytest

from aspire.denoising import adaptive_support
from aspire.image import Image
from aspire.source import ArrayImageSource
from aspire.utils import circ, gaussian_2d, inverse_r

logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "saved_test_data")


class AdaptiveSupportTest(TestCase):
    def setUp(self):

        self.size = size = 1025
        self.sigma = sigma = 128
        self.n_disc = n_disc = 10

        # Create 2D Gaussian as initial array.
        discs = np.tile(
            gaussian_2d(size, sigma_x=sigma, sigma_y=sigma),
            (n_disc, 1, 1),
        )

        # Add varying radius of solid disc.
        #  The Fourier transform will yield Airy disc
        #  which has more interesting content in F space.
        for d in range(n_disc):
            discs[d] = discs[d] + circ(size, radius=(d + 1) ** 2)

        self.img_src = ArrayImageSource(Image(discs))

        # Reference thesholds
        self.references = {
            1: 0.68,
            2: 0.96,
            3: 0.999,  # slightly off
            self.size / (2 * self.sigma): 1,
        }

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

    def test_adaptive_support_inverse_r(self):
        """
        Test `inverse_r` function support in Real and Fourier space is similar.

        The Fourier transform of `inverse_r` should be similar to real space,
        so we can test the support is similar.
        """

        # Generate stack of inverse_r function images.
        size = 64
        imgs = np.tile(
            inverse_r(size),
            (self.n_disc, 1, 1),
        )
        # Centered Fourier Transform
        f_imgs = np.fft.fftshift(np.sqrt(np.abs(np.fft.fft2(imgs))))

        # Setup ImageSource like objects
        img_src = ArrayImageSource(Image(imgs))
        f_img_src = ArrayImageSource(Image(f_imgs))

        thresholds = list(self.references.values())

        for threshold in thresholds:
            _, r = adaptive_support(img_src, threshold)
            _, rf = adaptive_support(f_img_src, threshold)
            # Test support is similar between original and transformed
            self.assertTrue(abs(r - rf) / r < 0.2)

    def test_adaptive_support_F(self):
        """
        Test Fourier support of Gaussian relates to normal distribution.
        """

        # Generate stack of 2D Gaussian images.
        imgs = np.tile(
            gaussian_2d(self.size, sigma_x=1 / self.sigma, sigma_y=1 / self.sigma),
            (self.n_disc, 1, 1),
        )

        # Setup ImageSource like objects
        img_src = ArrayImageSource(Image(imgs))

        thresholds = list(self.references.values())

        for threshold in thresholds:
            c, _ = adaptive_support(img_src, threshold)
            # Assert Fourier support is close to normal (doubled for sym).
            self.assertTrue(abs(2 * c - threshold) / threshold < 0.01)

    def test_adaptive_support_gaussian_circ(self):
        """
        Test against known Gaussian + circ.
        """

        # for one, two, three, inf standard deviations (one sided)
        for stddevs, threshold in self.references.items():
            # Real support should be ~ 1, 2, 3 times sigma
            c, r = adaptive_support(self.img_src, threshold)
            # Check closer to this threshold than next
            logger.info(f"{c} {r} = adaptive_support(..., {threshold})")
            self.assertTrue(np.abs(r / self.sigma - stddevs) < 0.5)
