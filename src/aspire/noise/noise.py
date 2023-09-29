import abc
import logging

import numpy as np

from aspire.image import Image
from aspire.image.xform import Xform
from aspire.numeric import fft, xp
from aspire.operators import (
    ArrayFilter,
    BlueFilter,
    PinkFilter,
    PowerFilter,
    ScalarFilter,
)
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
        _im = im.asnumpy().copy()

        for i, idx in enumerate(indices):
            # Note: The following random seed behavior is directly taken from MATLAB Cov3D code.
            random_seed = self.seed + 191 * (idx + 1)
            im_s = randn(2 * im.resolution, 2 * im.resolution, seed=random_seed)
            # Use numpy because im_s and im are different image sizes
            im_s = Image(im_s).filter(self.noise_filter).asnumpy()[0]
            _im[i] += im_s[: im.resolution, : im.resolution]

        return Image(_im)

    @abc.abstractproperty
    def noise_var(self):
        """
        Concrete implementations are expected to provide a method that
        returns the noise variance for the NoiseAdder.

        Authors of `NoiseAdder`s are encouraged to consider any relevant
        methods of calculating noise variance from theory.
        """


class CustomNoiseAdder(NoiseAdder):
    """
    Instantiates a NoiseAdder using the provided `noise_filter`.
    """

    @property
    def noise_var(self, res=512):
        """
        Return noise variance.

        CustomNoiseAdder will estimate noise_var using the `noise_filter`.

        :param res: Resolution to use when evaluating noise filter, default 512.
        :returns: Noise variance estimated at `res`.
        """
        # Take mean of user provided _noise_filter, before the PowerFilter is applied.
        return np.mean(self._noise_filter.evaluate_grid(res))


class WhiteNoiseAdder(NoiseAdder):
    """
    A Xform that adds white noise, optionally passed through a Filter
    object, to all incoming images.
    """

    # TODO, check if we can change seed and/or why not.
    def __init__(self, var, seed=0):
        """
        Return a `WhiteNoiseAdder` instance from `var` and using `seed`.

        :param var: Target noise variance.
        :param seed: Optinally provide a random seed used to generate white noise.
        """

        self.signal_power = None  # Used with `from_snr`
        self.requires_signal_power = False  # Used with `from_snr`
        self.noise_var = var
        self.seed = seed
        # When we know the var, complete building the filter.
        if var is not None:
            self._build()
        else:
            # Otherwise, we will flag that we require signal power.
            # Assigning `signal_power` later will complete builing
            # filter.
            self.requires_signal_power = True

    def _build(self):
        """
        Builds underlying Filter for this NoiseAdder.
        """
        super().__init__(
            noise_filter=ScalarFilter(dim=2, value=self.noise_var), seed=self.seed
        )

    @classmethod
    def from_snr(cls, snr, signal_power=None, seed=0):
        """
        Generates a WhiteNoiseAdder configured to produce a target
        signal to noise ratio.

        When `signal_power` is not provided, `requires_signal_power`
        attribute will be set.  Consumers can check this attribute and
        set `signal_power` as required. Setting `signal_power` should
        then complete building the filter.

        :param snr: Desired signal to noise ratio of
            the returned source.
        :param signal_power: Optional, if the signal power is known.
        :param seed: Optionally provide a random seed used to generate white noise.
        """

        noise_adder = cls(var=None, seed=seed)
        # signal_power.setter will use `_snr` to compute the noise
        # variance.
        noise_adder._snr = snr

        # `signal_power.setter` should complete _build when provided
        # `signal_power` is not None
        noise_adder.signal_power = signal_power

        return noise_adder

    def __str__(self):
        return f"{self.__class__.__name__} with variance={self._noise_var}"

    def __repr__(self):
        return f"{self.__class__.__name__}(var={self._noise_var}, seed={self.seed})"

    @property
    def noise_var(self):
        """
        Returns noise variance.

        Note in this white noise case, noise variance is known,
        because the `WhiteNoiseAdder` was instantied with an explicit variance.
        """
        return self._noise_var

    @noise_var.setter
    def noise_var(self, v):
        self._noise_var = v

    @property
    def signal_power(self):
        return self._signal_power

    @signal_power.setter
    def signal_power(self, p):
        self._signal_power = p
        if p is not None:
            self.requires_signal_power = False
            self.noise_var = p / self._snr
            self._build()


class BlueNoiseAdder(WhiteNoiseAdder):
    """
    NoiseAdder where noise power increases with frequency.
    """

    def _build(self):
        """
        Builds underlying Filter for this NoiseAdder.
        """

        # Call the __init__ from parent of WhiteNoiseAdder.
        super(WhiteNoiseAdder, self).__init__(
            noise_filter=BlueFilter(var=self.noise_var), seed=self.seed
        )


class PinkNoiseAdder(WhiteNoiseAdder):
    """
    NoiseAdder where noise power decreases with frequency.
    """

    def _build(self):
        """
        Builds underlying Filter for this NoiseAdder.
        """

        # Call the __init__ from parent of WhiteNoiseAdder.
        super(WhiteNoiseAdder, self).__init__(
            noise_filter=PinkFilter(var=self.noise_var), seed=self.seed
        )


class NoiseEstimator:
    """
    Noise Estimator base class.
    """

    def __init__(self, src, bgRadius=1, batchSize=512):
        """
        Any additional args/kwargs are passed on to the Source's 'images' method

        :param src: A Source object which can give us images on demand
        :param bgRadius: The radius of the disk whose complement is used to estimate the noise.
            Radius is relative proportion, where `1` represents
            the radius of disc inscribing a `(src.L, src.L)` image.
        :param batchSize:  The size of the batches in which to compute the variance estimate.
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
            second_moment += np.sum(np.abs(images_masked) ** 2) / _denominator
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
            noise_psd_est += np.sum(np.abs(im_masked_f) ** 2, axis=0) / _denominator

        mid = self.src.L // 2
        noise_psd_est[mid, mid] -= mean_est**2

        return noise_psd_est
