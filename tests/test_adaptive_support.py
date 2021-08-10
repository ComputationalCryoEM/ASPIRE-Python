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
        self.sigma = sigma = 16
        self.n_disc = 10

        # Reference thresholds. Since we're integrating 2 * r * exp(-r ** 2 /
        # (2 * sigma ** 2)), the thresholds corresponding to one, two, and
        # three standard deviations are the following.
        self.references = {
            1: 1 - np.exp(-1 / 2),
            2: 1 - np.exp(-2 ** 2 / 2),
            3: 1 - np.exp(-3 ** 2/ 2)
        }

    def testAdaptiveSupportBadThreshold(self):
        """
        Method should raise meaningful error when passed unreasonable thresholds.
        """

        discs = np.empty((self.size, self.size))  # Intentional Dummy Data
        img_src = ArrayImageSource(Image(discs))

        with pytest.raises(ValueError, match=r"Given energy_threshold.*"):
            _ = adaptive_support(img_src, -0.5)

        with pytest.raises(ValueError, match=r"Given energy_threshold.*"):
            _ = adaptive_support(img_src, 9000)

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

    @pytest.mark.xfail(reason="Adaptive Support Algorithm may be incorrect.")
    def test_adaptive_support_inverse_r(self):
        """
        Test `inverse_r` function support in Real and Fourier space is similar.

        The Fourier transform of `inverse_r` should be similar to real space,
        so we can test the support is similar.
        """

        # Generate stack of inverse_r function images.
        imgs = np.tile(
            inverse_r(self.size),
            (self.n_disc, 1, 1),
        )

        # Setup ImageSource like objects
        img_src = ArrayImageSource(Image(imgs))

        thresholds = list(self.references.values())

        for threshold in thresholds:
            c, r = adaptive_support(img_src, threshold)
            # Test support is similar between original and transformed
            logger.info(
                f"Threshold {threshold} c {c} cr { c * self.size} R {r} diff {abs(r - c * self.size) / r }"
            )
            self.assertTrue(abs(r - c * self.size) / r < 0.2)

    def test_adaptive_support_F(self):
        """
        Test Fourier support of Gaussian relates to normal distribution.
        """

        # Generate stack of 2D Gaussian images.
        imgs = np.tile(
            gaussian_2d(self.size, sigma_x=self.sigma, sigma_y=self.sigma),
            (self.n_disc, 1, 1),
        )

        # Setup ImageSource like objects
        img_src = ArrayImageSource(Image(imgs))

        thresholds = list(self.references.values())

        for ref, threshold in self.references.items():
            c, R = adaptive_support(img_src, threshold)

            # Assert spatial support is close to normal.
            R_true = ref * self.sigma

            # Standard deviation in Fourier space is given by 1/(2 * pi *
            # sigma). This can be obtained by applying the Poisson summation
            # formula to the continuous FT which gives that the discrete FT is
            # well approximated by a Gaussian with that particular standard
            # deviation.
            c_true = ref / (2 * np.pi * self.sigma)

            # Since we're dealing with the square of the Gaussian, this
            # effectively divides the sigmas by sqrt(2).
            R_true /= np.sqrt(2)
            c_true /= np.sqrt(2)

            # Accuracy is not perfect, but within 5% if sigma is in the right
            # range (too small, R is inaccurate; too big, c is inaccurate.
            self.assertTrue(abs(R - R_true) / R_true < 0.05)
            self.assertTrue(abs(c - c_true) / c_true < 0.05)

    def test_adaptive_support_gaussian_circ(self):
        """
        Test against known Gaussian + circ.
        """

        # Create 2D Gaussian as initial array.
        discs = np.tile(
            gaussian_2d(self.size, sigma_x=self.sigma, sigma_y=self.sigma),
            (self.n_disc, 1, 1),
        )

        # Add varying radius of solid disc.
        #  The Fourier transform will yield Airy disc
        #  which has more interesting content in F space.
        for d in range(self.n_disc):
            discs[d] = discs[d] + circ(self.size, radius=(d + 1) ** 2)

        img_src = ArrayImageSource(Image(discs))

        # for one, two, three, inf standard deviations (one sided)
        for stddevs, threshold in self.references.items():
            # Real support should be ~ 1, 2, 3 times sigma
            c, r = adaptive_support(img_src, threshold)
            # Check closer to this threshold than next
            logger.info(f"{c} {r} = adaptive_support(..., {threshold})")
            self.assertTrue(np.abs(r / self.sigma - stddevs) < 0.5)
