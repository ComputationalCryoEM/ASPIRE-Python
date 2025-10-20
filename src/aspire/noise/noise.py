import abc
import logging
from functools import cached_property

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
from aspire.utils import gaussian_window, grid_2d, randn, trange

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

        return Image(_im, pixel_size=im.pixel_size)

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

    def __init__(self, src, bg_radius=1, batch_size=512):
        """
        Any additional args/kwargs are passed on to the Source's 'images' method

        :param src: A Source object which can give us images on demand
        :param bg_radius: The radius of the disk whose complement is used to estimate the noise.
            Radius is relative proportion, where `1` represents
            the radius of disc inscribing a `(src.L, src.L)` image.
        :param batch_size:  The size of the batches in which to compute the variance estimate.
        """
        self.src = src
        self.dtype = self.src.dtype
        self.bg_radius = bg_radius
        self.batch_size = batch_size

    @cached_property
    def filter(self):
        """
        Property returning `Filter` object for this noise estimator.
        This property will be computed and cached on first call.

        :return: `Filter` object.
        """

        return self._create_filter()

    @abc.abstractmethod
    def estimate(self):
        """
        :return: The estimated noise variance of the images.
        """

    @abc.abstractmethod
    def _create_filter(self):
        """
        Private method for computing and returning `Filter` object.
        """


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

    def _create_filter(self):
        """
        :return: The estimated noise power spectral distribution (PSD) of the images in the form of a filter object.
        """
        logger.info(f"Determining Noise variance in batches of {self.batch_size}")
        noise_variance = self._estimate_noise_variance()
        logger.info(f"Noise variance = {noise_variance}")
        return ScalarFilter(dim=2, value=noise_variance)

    def _estimate_noise_variance(self):
        """
        :return: The estimated noise variance of the images in the Source used to create this estimator.
        TODO: How's this initial estimate of variance different from the 'estimate' method?
        """
        # Run estimate using saved parameters
        g2d = grid_2d(self.src.L, indexing="yx", dtype=self.dtype)
        mask = g2d["r"] >= self.bg_radius

        first_moment = 0
        second_moment = 0
        for i in trange(0, self.src.n, self.batch_size):
            images = self.src.images[i : i + self.batch_size].asnumpy()
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

    def _create_filter(self):
        """
        :return: The estimated noise power spectral distribution (PSD) of the images in the form of a filter object.
        """
        return ArrayFilter(self._estimate_noise_psd())

    def _estimate_noise_psd(self):
        """
        :return: The estimated noise variance of the images in the Source used to create this estimator.
        TODO: How's this initial estimate of variance different from the 'estimate' method?
        """
        # Run estimate using saved parameters
        g2d = grid_2d(self.src.L, indexing="yx", dtype=self.dtype)
        mask = g2d["r"] >= self.bg_radius

        mean_est = 0
        noise_psd_est = np.zeros((self.src.L, self.src.L)).astype(self.src.dtype)
        for i in trange(0, self.src.n, self.batch_size):
            images = self.src.images[i : i + self.batch_size].asnumpy()
            images_masked = images * mask

            _denominator = self.src.n * np.sum(mask)
            mean_est += np.sum(images_masked) / _denominator
            im_masked_f = xp.asnumpy(fft.centered_fft2(xp.asarray(images_masked)))
            noise_psd_est += np.sum(np.abs(im_masked_f) ** 2, axis=0) / _denominator

        mid = self.src.L // 2
        noise_psd_est[mid, mid] -= mean_est**2

        return noise_psd_est


class LegacyNoiseEstimator(NoiseEstimator):
    """
    Isotropic noise estimator ported from MATLAB `cryo_noise_estimation`.
    """

    def __init__(
        self, src, bg_radius=None, max_d=None, batch_size=512, normalize_psd=False
    ):
        """
        Given an `ImageSource`, prepares for estimation of noise spectrum.

        Estimate is delayed and computed on first access of `filter` attribute.

        :param src: A `ImageSource` object.
        :param bg_radius: The radius of the disk whose complement is used to estimate the noise.
            Radius is relative proportion, where `1` represents
            the radius of disc inscribing a `(src.L, src.L)` image.
            Default of `None` uses `(np.floor(src.L / 2) - 1) / L`
        :param max_d: Max computed correlation distance as a proportion of `src.L`.
            Default of `None` uses `np.floor(src.L/3) / L`.
        :param batch_size:  The size of the batches in which to compute the variance estimate.
        :param normalize_psd: Optionally normalize PSD in way that reproduces MATLAB intermediates.
            Only useful if utilizing the `PSD` for applications outside of built-in legacy whitening.
        """

        if bg_radius is None:
            bg_radius = (np.floor(src.L / 2) - 1) / src.L

        super().__init__(src=src, bg_radius=bg_radius, batch_size=batch_size)

        self.max_d = max_d
        if self.max_d is None:
            self.max_d = np.floor(src.L / 3) / src.L

        self.normalize_psd = bool(normalize_psd)

    def estimate(self):
        """
        :return: The estimated noise variance of the images.
        """

        return np.mean(self.filter.evaluate_grid(self.src.L))

    def _create_filter(self):
        """
        :return: The estimated noise power spectral distribution (PSD) of the images in the form of a filter object.
        """
        return ArrayFilter(self._estimate_noise_psd())

    def _estimate_noise_psd(self):
        """
        :return: The estimated noise variance of the images in the Source used to create this estimator.
        """
        # Setup parameters
        samples_idx = grid_2d(self.src.L, normalized=False, shifted=True)["r"] >= (
            self.bg_radius * self.src.L
        )
        max_d_pixels = round(self.max_d * self.src.L)

        psd = self._estimate_power_spectrum_distribution_2d(
            self.src.images,
            samples_idx,
            max_d_pixels,
            batch_size=self.batch_size,
            normalize_psd=self.normalize_psd,
        )[0]

        return psd

    @staticmethod
    def _estimate_power_spectrum_distribution_1d(
        images, samples_idx, max_d=None, batch_size=512
    ):
        """
        Estimate the 1D isotropic autocorrelation function of `images`.
        The samples to use in each image are given by `samples_idx` mask.
        The correlation is computed up to a maximal distance of `max_d`.

        Port of MATLAB `cryo_epsdR`.

        :param images: `Image` instance
        :param samples_idx: Boolean mask shaped `(L,L)`.
        :param max_d: Max computed correlation distance in pixels.
           Default of `None` yields `np.floor(L / 3)`.
        :param batch_size:  The size of the batches in which to compute the variance estimate.
        :return:
            - Radial PSD array of shape
            - Distances map array
            - Count of nonzero correlations array
        """

        n_img = images.n_images
        L = samples_idx.shape[-1]
        batch_size = min(batch_size, n_img)

        # Correlations more than `max_d` pixels apart are not computed.
        if max_d is None:
            max_d = np.floor(L / 3)
        if max_d > L - 1:
            logger.info(
                f"`max_d` value {max_d}greater than number of image pixels {L}, clipping to {L-1}."
            )
        max_d = int(min(max_d, L - 1))

        # Compute distances
        # Note grid_2d['r'] is not used because we want an integer grid directly;
        #   yields integer dists (radius**2) values.
        X, Y = xp.mgrid[0 : max_d + 1, 0 : max_d + 1]
        dsquare = X * X + Y * Y
        uniq_dsquare = xp.sort(xp.unique(dsquare[dsquare <= max_d**2]))
        x = xp.sqrt(uniq_dsquare)  # actual distances

        # corrs[i] is the sum of all x[j]x[j+d] where d = x[i]
        corrs = xp.zeros_like(uniq_dsquare, dtype=np.float64)
        # corrcount[i] is the number of pairs summed in corr[i]
        corrcount = xp.zeros_like(uniq_dsquare, dtype=np.int64)

        # distmap maps [i,j] to k where uniq_dsquare[k] = i**2 + j**2.
        #   -1 indicates distance is larger than max_d
        distmap = xp.full(shape=dsquare.shape, fill_value=-1)

        # This differs from the MATLAB code, avoids `bisect`.
        for i, d in enumerate(uniq_dsquare):
            inds = dsquare == d  # locations having distance `d`
            distmap[inds] = i  # assign index into uniq_dsquare `i`
        # From here on, distmap will be accessed with flat indices
        distmap = distmap.flatten()
        valid_dists = xp.argwhere(distmap != -1)

        # Compute Ncorr using a constant unit image.
        mask = xp.zeros((L, L))
        mask[samples_idx] = 1
        buf_padded = xp.zeros((batch_size, 2 * L + 1, 2 * L + 1))  # pad
        buf_padded[0, :L, :L] = mask
        # MATLAB code internally detects/implicitly casts,
        #   we explicitly call rfft2/irfft2.
        fbuf_padded = fft.rfft2(buf_padded[0])
        n_mask_pairs = fft.irfft2(
            fbuf_padded * fbuf_padded.conj(), s=buf_padded.shape[1:]
        )
        n_mask_pairs = n_mask_pairs[: max_d + 1, : max_d + 1]  # crop
        n_mask_pairs = xp.round(n_mask_pairs)

        samples = xp.zeros((batch_size, L, L))
        buf_padded[0, :, :] = 0  # reset buf_padded
        for start in trange(
            0, n_img, batch_size, desc="Processing image autocorrelations"
        ):
            end = min(n_img, start + batch_size)
            count = end - start
            # Pack masked `sample_idx` pixels from `images` batch into `samples`
            samples[:count, samples_idx] = images[start:end].asnumpy()[:, samples_idx]
            # Optimization note: We could also compute the noise
            # energy estimate used later at this time to avoid looping
            # over images twice.

            # Compute non-periodic autocorrelation
            buf_padded[:count, :L, :L] = samples[:count]  # pad
            # MATLAB code internally detects/implicitly casts,
            #   we explicitly call rfft2/irfft2.
            fbuf_padded = fft.rfft2(buf_padded[:count])
            s = fft.irfft2(fbuf_padded * fbuf_padded.conj(), s=buf_padded.shape[1:])
            s = s[:, 0 : max_d + 1, 0 : max_d + 1]  # crop

            # Accumulate all autocorrelation values R[k1,k2] such that
            # k1**2 + k2**2 = dist (all autocorrelations of a certain distance).
            s = xp.sum(s, axis=0).flatten()
            _n_mask_pairs = n_mask_pairs.flatten() * count
            for d in valid_dists:
                idx = distmap[d]
                corrs[idx] = corrs[idx] + s[d]
                corrcount[idx] = corrcount[idx] + _n_mask_pairs[d]

        # Values of isotropic autocorrelation function
        # R[i] is value of ACF at distance x[i]
        # Remove distances which had no samples
        idx = xp.where(corrcount != 0)
        R = corrs[idx] / corrcount[idx]
        x = xp.asnumpy(x[idx])
        cnt = corrcount[idx]

        R = xp.asnumpy(R)
        cnt = xp.asnumpy(cnt)

        return R, x, cnt

    @staticmethod
    def _estimate_power_spectrum_distribution_2d(
        images, samples_idx, max_d=None, batch_size=512, normalize_psd=False
    ):
        """
        Estimate the 2D isotropic power spectrum of `images`.
        The samples to use in each image are given by `samples_idx` mask.
        The correlation is computed up to a maximal distance of `max_d`.

        Port of MATLAB `cryo_epsdS`.

        :param images: Images instance
        :param samples_idx: Boolean mask shaped (L,L).
        :param max_d: Max computed correlation distance in pixels.
           Default of `None` yields `np.floor(L / 3)`.
        :param batch_size:  The size of the batches in which to compute the variance estimate.
        :normalize_psd:  Optionally normalize returned PSD.
            Disabled by default because it will typically be
            renormalized later in preperation for the convolution
            application in `Image.legacy_whiten`.
            Enable to reproduce legacy PSD.
        :return:
            - 2D PSD array
            - Radial PSD array
            - Distances map array
            - Count of nonzero correlations array
        """

        n_img = images.n_images
        L = samples_idx.shape[-1]
        batch_size = min(batch_size, n_img)

        # Correlations more than `max_d` pixels apart are not computed.
        if max_d is None:
            max_d = np.floor(L / 3)
        if max_d > L - 1:
            logger.info(
                f"`max_d` value {max_d}greater than number of image pixels {L}, clipping to {L-1}."
            )
        max_d = int(min(max_d, L - 1))

        R, x, _ = LegacyNoiseEstimator._estimate_power_spectrum_distribution_1d(
            images=images, samples_idx=samples_idx, max_d=max_d, batch_size=batch_size
        )
        _R = xp.asarray(R)  # Migrate to GPU for assignments below

        # Use the 1D autocorrelation estimated above to populate an
        # array of the 2D isotropic autocorrelction. This
        # autocorrelation is later Fourier transformed to get the
        # power spectrum.
        R2 = xp.zeros((2 * L - 1, 2 * L - 1), dtype=np.float64)

        X, Y = xp.mgrid[-L + 1 : L, -L + 1 : L]
        dists = X * X + Y * Y
        uniq_dsquare = xp.sort(xp.unique(dists[dists <= max_d**2]))
        for i, d in enumerate(uniq_dsquare):
            idx = dists == d
            R2[idx] = _R[i]

        # Window the 2D autocorrelation and Fourier transform it to get the power
        # spectrum. Always use the Gaussian window, as it has positive Fourier
        # transform.
        w = xp.asarray(gaussian_window(L, max_d))
        P2 = fft.centered_fft2(R2 * w)
        if (err := xp.linalg.norm(P2.imag) / xp.linalg.norm(P2)) > 1e-12:
            logger.warning(f"Large imaginary components in P2 {err}.")
        P2 = P2.real

        if normalize_psd:
            # Normalize the power spectrum P2. The power spectrum is normalized such
            # that its energy is equal to the average energy of the noise samples used
            # to estimate it.

            E = 0.0  # Total energy of the noise samples used to estimate the power spectrum.
            n_samples_per_img = int(np.count_nonzero(samples_idx))
            samples = xp.zeros((batch_size, n_samples_per_img), dtype=np.float64)
            for start in trange(
                0, n_img, batch_size, desc="Estimating image noise energy"
            ):
                end = min(n_img, start + batch_size)
                cnt = end - start

                samples[:cnt] = xp.asarray(images[start:end].asnumpy()[:, samples_idx])
                E += xp.sum(
                    (samples[:cnt] - xp.mean(samples[:cnt], axis=-1).reshape(cnt, 1))
                    ** 2
                )

            # Mean energy of the noise samples
            meanE = E / (n_samples_per_img * n_img)

            # Normalize P2 such that its mean energy is preserved and is equal to
            # meanE, that is, mean(P2)==meanE. That way the mean energy does not
            # go down if the number of pixels is artifically changed (say be
            # upsampling, downsampling, or cropping). Note that P2 is already in
            # units of energy, and so the total energy is given by sum(P2) and
            # not by norm(P2).
            P2 = P2 / xp.sum(P2) * meanE * P2.size

        # Check that P2 has no negative values.
        # Due to the truncation of the Gaussian window, we get small negative
        # values. So unless they are very big, we just ignore them.
        negidx = P2 < 0
        if xp.count_nonzero(negidx):
            maxnegerr = xp.max(xp.abs(P2[negidx]))
            logger.debug(f"Maximal negative P2 value = {maxnegerr}")
            if maxnegerr > 1e-2:
                negnorm = xp.linalg.norm(P2[negidx])
                logger.warning(
                    f"Power spectrum P2 has non trivial negative values with energy {negnorm}."
                )
            P2[negidx] = 0  # zero out negative estimates

        P2 = xp.asnumpy(P2)
        R2 = xp.asnumpy(R2)
        # R, x already on host
        return P2, R, R2, x
