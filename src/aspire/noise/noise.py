import logging

import numpy as np

from aspire.numeric import fft, xp
from aspire.operators import ArrayFilter, ScalarFilter
from aspire.utils.coor_trans import grid_2d

logger = logging.getLogger(__name__)


# TODO: Implement correct hierarchy and DRY


class NoiseEstimator:
    """
    Noise Estimator base class.
    """

    def __init__(self, src, bgRadius=1, batchSize=512):
        """
        Any additional args/kwargs are passed on to the Source's 'images' method
        :param src: A Source object which can give us images on demand
        :param bgRadius: The radius of the disk whose complement is used to estimate the noise.
        :param batchSize:  The size of the batches in which to compute the variance estimate
        """

        self.src = src
        self.dtype = self.src.dtype
        self.L = src.L
        self.n = src.n
        self.bgRadius = bgRadius
        self.batchSize = batchSize

        self.filter = self._create_filter()

    def estimate(self):
        """
        :return: The estimated noise variance of the images.
        """
        raise NotImplementedError("Subclasses implement the `estimate` method.")


class WhiteNoiseEstimator(NoiseEstimator):
    """
    White Noise Estimator.
    """

    def estimate(self):
        """
        :return: The estimated noise variance of the images.
        """
        return self.filter.evaluate(np.zeros((2, 1), dtype=self.dtype)).item()

    def _create_filter(self, noise_variance=None):
        """
        :param noise_variance: Noise variance of images
        :return: The estimated noise power spectral distribution (PSD) of the images in the form of a filter object.
        """
        if noise_variance is None:
            logger.info(f"Determining Noise variance in batches of {self.batchSize}")
            noise_variance = self._estimate_noise_variance()
            logger.info(f"Noise variance = {noise_variance}")
        return ScalarFilter(dim=2, value=noise_variance)

    def _estimate_noise_variance(self):
        """
        Any additional arguments/keyword-arguments are passed on to the Source's 'images' method
        :return: The estimated noise variance of the images in the Source used to create this estimator.
        TODO: How's this initial estimate of variance different from the 'estimate' method?
        """
        # Run estimate using saved parameters
        g2d = grid_2d(self.L, dtype=self.dtype)
        mask = g2d["r"] >= self.bgRadius

        first_moment = 0
        second_moment = 0
        for i in range(0, self.n, self.batchSize):
            images = self.src.images(start=i, num=self.batchSize).asnumpy()
            images_masked = images * mask

            _denominator = self.n * np.sum(mask)
            first_moment += np.sum(images_masked) / _denominator
            second_moment += np.sum(np.abs(images_masked ** 2)) / _denominator
        return second_moment - first_moment ** 2


class AnisotropicNoiseEstimator(NoiseEstimator):
    """
    Anisotropic White Noise Estimator.
    """

    def estimate(self):
        """
        :return: The estimated noise variance of the images.
        """
        return self.filter.evaluate(np.zeros((2, 1))).item()

    def _create_filter(self, noise_psd=None):
        """
        :param noise_psd: Noise PSD of images
        :return: The estimated noise power spectral distribution (PSD) of the images in the form of a filter object.
        """
        if noise_psd is None:
            noise_psd = self.estimate_noise_psd()
        return ArrayFilter(noise_psd)

    def estimate_noise_psd(self):
        """
        :return: The estimated noise variance of the images in the Source used to create this estimator.
        TODO: How's this initial estimate of variance different from the 'estimate' method?
        """
        # Run estimate using saved parameters
        g2d = grid_2d(self.L)
        mask = g2d["r"] >= self.bgRadius

        mean_est = 0
        noise_psd_est = np.zeros((self.L, self.L)).astype(self.src.dtype)
        for i in range(0, self.n, self.batchSize):
            images = self.src.images(i, self.batchSize).asnumpy()
            images_masked = images * mask

            _denominator = self.n * np.sum(mask)
            mean_est += np.sum(images_masked) / _denominator
            im_masked_f = xp.asnumpy(fft.centered_fft2(xp.asarray(images_masked)))
            noise_psd_est += np.sum(np.abs(im_masked_f ** 2), axis=0) / _denominator

        mid = self.L // 2
        noise_psd_est[mid, mid] -= mean_est ** 2

        return noise_psd_est


class IsotropicNoiseEstimator(AnisotropicNoiseEstimator):
    """
    Define a derived class of isotropic noise estimator
    """

    def estimate(self):
        """
        :return: The estimated noise variance of the images.
        """
        return self.filter.evaluate(np.zeros((1, 1))).item()

    def estimate_noise_psd(self):
        """
        :return: The estimated isotropic PSD for creating a filter.
        """
        # Estimate  anisotropic PSD
        g2d = grid_2d(self.L)
        mask = g2d["r"] >= self.bgRadius

        mean_est = 0
        noise_psd_est = np.zeros((self.L, self.L)).astype(self.src.dtype)
        for i in range(0, self.n, self.batchSize):
            images = self.src.images(i, self.batchSize).asnumpy()
            images_masked = images * mask

            _denominator = self.n * np.sum(mask)
            mean_est += np.sum(images_masked) / _denominator
            im_masked_f = xp.asnumpy(fft.centered_fft2(xp.asarray(images_masked)))
            noise_psd_est += np.sum(np.abs(im_masked_f ** 2), axis=0) / _denominator

        mid = self.L // 2
        noise_psd_est[mid, mid] -= mean_est ** 2

        # Estimate isotropic PSD
        iso_noise_psd = np.zeros(self.L)
        npoints = np.zeros(self.L)
        dr = 2 * np.max(g2d["r"]) / self.L
        # Calculate PSD at middle point
        mask = (g2d["r"] >= 0.0) & (g2d["r"] < dr)
        iso_noise_psd[mid] = np.sum(noise_psd_est[mask])
        npoints[mid] = np.count_nonzero(mask) * np.pi * dr ** 2
        # Calculate PSD at negative and positive values
        for i in range(1, mid + 1):
            mask = (g2d["r"] >= dr * i) & (g2d["r"] < dr * (i + 1))
            sum_psd = np.sum(noise_psd_est[mask])
            sum_npt = np.count_nonzero(mask) * 2 * np.pi * dr * i * dr
            if mid - i >= 0:
                iso_noise_psd[mid - i] = sum_psd
                npoints[mid - i] = sum_npt
            if mid + i < self.L:
                iso_noise_psd[mid + i] = sum_psd
                npoints[mid + i] = sum_npt
        iso_noise_psd = iso_noise_psd / npoints

        return iso_noise_psd
