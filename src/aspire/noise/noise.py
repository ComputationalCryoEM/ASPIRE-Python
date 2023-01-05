import abc
import logging

import numpy as np

from aspire.image import Image
from aspire.image.xform import Xform
from aspire.numeric import fft, xp
from aspire.operators import ArrayFilter, PowerFilter, ScalarFilter
from aspire.utils import grid_2d, randn, trange

logger = logging.getLogger(__name__)


# TODO: Implement correct hierarchy and DRY


class NoiseAdder(Xform):
    """
    Defines interface for `CustomNoiseAdder`s.
    """

    def __init__(self, noise_filter, seed=0):
        """
        Initialize the random state of this `NoiseAdder` using `noise_filter` and `seed`.

        `noise_filter` will be provided by the user or instantiated automatically by the subclass.

        :param seed: The random seed used to generate white noise.
        :param noise_filter: An `aspire.operators.Filter` object.
            `NoiseAdders` start by generating gaussian noise,
            then apply `noise_filter` to transform the noise.
            Note the `noise_filter` will be raised to the 1/2 power.
        """
        super().__init__()
        self.seed = seed
        self._noise_filter = noise_filter
        self.noise_filter = PowerFilter(noise_filter, power=0.5)

    def __repr__(self):
        return f"{self.__class__.__name__}(noise_filter={self._noise_filter}, seed={self.seed})"

    def __str__(self):
        return f"{self.__class__.__name__}"

    def _forward(self, im, indices):
        im = im.copy()

        for i, idx in enumerate(indices):
            # Note: The following random seed behavior is directly taken from MATLAB Cov3D code.
            random_seed = self.seed + 191 * (idx + 1)
            im_s = randn(2 * im.resolution, 2 * im.resolution, seed=random_seed)
            im_s = Image(im_s).filter(self.noise_filter)[0]
            im[i] += im_s[: im.resolution, : im.resolution]

        return im

    @abc.abstractproperty
    def noise_var(self):
        """
        Concrete implementations are expected to provide a method that
        returns the noise variance for the NoiseAdder.
        """


class CustomNoiseAdder(NoiseAdder):
    """
    Instantiates a NoiseAdder using the provided `noise_filter`.
    """

    @property
    def noise_var(self):
        """
        Return noise variance.

        CustomNoiseAdder will estimate noise_var by taking a sample of the noise.

        If you require tuning the noise_var sampling, see `get_noise_var`.
        """
        return self.get_noise_var()

    def get_noise_var(self, sample_n=100, sample_res=128):
        """
        Return noise variance.

        CustomNoiseAdder will estimate noise_var by taking a sample of the noise.

        It is highly encouraged that authors of `CustomNoiseAdder`s consider
        any theoretically superior methods of calculating noise variance,
        or test that this method's default values are satisfactory for their
        implementation.

        :sample_n: Number of images to sample.
        :sample_res: Resolution of sample (noise) images.
        :returns: Noise Variance.
        """
        im_zeros = Image(np.zeros((sample_n, sample_res, sample_res)))
        im_noise_sample = self._forward(im_zeros, range(sample_n))
        return np.var(im_noise_sample.asnumpy())


class WhiteNoiseAdder(NoiseAdder):
    """
    A Xform that adds white noise, optionally passed through a Filter object, to all incoming images.
    """

    # TODO, check if we can change seed and/or why not.
    def __init__(self, var, seed=0):
        """
        Return a `WhiteNoiseAdder` instance from `var` and using `seed`.

        :param var: Target noise variance.
        :param seed: Optinally provide a random seed used to generate white noise.
        """
        self._noise_var = var
        super().__init__(noise_filter=ScalarFilter(dim=2, value=var), seed=seed)

    def __str__(self):
        return f"{self.__class__.__name__} with variance={self._noise_var}"

    def __repr__(self):
        return f"{self.__class__.__name__}(var={self._noise_var}, seed={self.seed})"

    @property
    def noise_var(self):
        """
        Returns noise variance.

        Note in this white noise case noise variance is known,
        because the `WhiteNoiseAdder` was instantied with an explicit variance.
        """
        return self._noise_var


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
        # WhiteNoiseEstimator.filter is a ScalarFilter,
        #   so we only evaluate for the zero frequencies.
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
        g2d = grid_2d(self.src.L, indexing="yx", dtype=self.dtype)
        mask = g2d["r"] >= self.bgRadius

        first_moment = 0
        second_moment = 0
        for i in trange(0, self.src.n, self.batchSize):
            images = self.src.images[i : i + self.batchSize].asnumpy()
            images_masked = images * mask

            _denominator = self.src.n * np.sum(mask)
            first_moment += np.sum(images_masked) / _denominator
            second_moment += np.sum(np.abs(images_masked**2)) / _denominator
        return second_moment - first_moment**2


class AnisotropicNoiseEstimator(NoiseEstimator):
    """
    Anisotropic White Noise Estimator.
    """

    def estimate(self):
        """
        :return: The estimated noise variance of the images.
        """

        # AnisotropicNoiseEstimator.filter is an ArrayFilter.
        #   We average the variance over all frequencies,

        return np.mean(self.filter.evaluate_grid(self.src.L))

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
        g2d = grid_2d(self.src.L, indexing="yx", dtype=self.dtype)
        mask = g2d["r"] >= self.bgRadius

        mean_est = 0
        noise_psd_est = np.zeros((self.src.L, self.src.L)).astype(self.src.dtype)
        for i in trange(0, self.src.n, self.batchSize):
            images = self.src.images[i : i + self.batchSize].asnumpy()
            images_masked = images * mask

            _denominator = self.src.n * np.sum(mask)
            mean_est += np.sum(images_masked) / _denominator
            im_masked_f = xp.asnumpy(fft.centered_fft2(xp.asarray(images_masked)))
            noise_psd_est += np.sum(np.abs(im_masked_f**2), axis=0) / _denominator

        mid = self.src.L // 2
        noise_psd_est[mid, mid] -= mean_est**2

        return noise_psd_est
