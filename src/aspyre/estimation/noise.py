import logging
import numpy as np

from aspyre.utils.math import grid_2d
from aspyre.utils.fft import centered_fft2
from aspyre.imaging.filters import ScalarFilter, ArrayFilter

logger = logging.getLogger(__name__)


# TODO: Implement correct hierarchy and DRY

class NoiseEstimator:
    pass


class WhiteNoiseEstimator(NoiseEstimator):
    def __init__(self, src, bgRadius=1, batchSize=512):
        """
        A White Noise Estimator
        Any additional args/kwargs are passed on to the Source's 'images' method
        :param src: A Source object which can give us images on demand
        :param bgRadius: The radius of the disk whose complement is used to estimate the noise.
        :param batchSize:  The size of the batches in which to compute the variance estimate
        """
        self.src = src
        self.L = src.L
        self.n = src.n
        self.bgRadius = bgRadius
        self.batchSize = batchSize

        self.filter = self._create_filter()

    def estimate(self):
        """
        :return: The estimated noise variance of the images.
        """
        return np.asscalar(self.filter.evaluate(np.zeros((2, 1))))

    def _create_filter(self, noise_variance=None):
        """
        :param noise_variance: Noise variance of images
        :return: The estimated noise power spectral distribution (PSD) of the images in the form of a filter object.
        """
        if noise_variance is None:
            logger.info(f'Determining Noise variance in batches of {self.batchSize}')
            noise_variance = self._estimate_noise_variance()
            logger.info(f'Noise variance = {noise_variance}')
        return ScalarFilter(dim=2, value=noise_variance)

    def _estimate_noise_variance(self):
        """
        Any additional arguments/keyword-arguments are passed on to the Source's 'images' method
        :return: The estimated noise variance of the images in the Source used to create this estimator.
        TODO: How's this initial estimate of variance different from the 'estimate' method?
        """
        # Run estimate using saved parameters
        g2d = grid_2d(self.L)
        mask = g2d['r'] >= self.bgRadius

        first_moment = 0
        second_moment = 0
        for i in range(0, self.n, self.batchSize):
            images = self.src.images(start=i, num=self.batchSize)
            images_masked = (images * np.expand_dims(mask, 2))

            _denominator = self.n * np.sum(mask)
            first_moment += np.sum(images_masked) / _denominator
            second_moment += np.sum(np.abs(images_masked**2)) / _denominator
        return second_moment - first_moment**2


class AnisotropicNoiseEstimator(NoiseEstimator):
    def __init__(self, src, bgRadius=1, batchSize=512):
        """
        A White Noise Estimator
        :param src: A Source object which can give us images on demand
        :param bgRadius: The radius of the disk whose complement is used to estimate the noise.
        :param batchSize:  The size of the batches in which to compute the variance estimate
        TODO: A base class NoiseEstimator should be instantiated using any object that can furnish images.
        """
        self.src = src
        self.L = src.L
        self.n = src.n
        self.bgRadius = bgRadius
        self.batchSize = batchSize

        self.filter = self._create_filter()

    def estimate(self):
        """
        :return: The estimated noise variance of the images.
        """
        return np.asscalar(self.filter.evaluate(np.zeros((2, 1))))

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
        mask = g2d['r'] >= self.bgRadius

        mean_est = 0
        noise_psd_est = np.zeros((self.L, self.L)).astype(self.src.dtype)
        for i in range(0, self.n, self.batchSize):
            images = self.src.images(i, self.batchSize)
            images_masked = (images * np.expand_dims(mask, 2))

            _denominator = self.n * np.sum(mask)
            mean_est += np.sum(images_masked) / _denominator
            im_masked_f = centered_fft2(images_masked)
            noise_psd_est += np.sum(np.abs(im_masked_f**2), axis=2) / _denominator

        mid = self.L // 2
        noise_psd_est[mid, mid] -= mean_est**2

        return noise_psd_est
