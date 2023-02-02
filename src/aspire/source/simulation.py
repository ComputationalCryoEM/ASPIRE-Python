import copy
import logging

import numpy as np
from scipy.linalg import eigh, qr
from sklearn.metrics import mean_squared_error

from aspire.image import Image
from aspire.noise import NoiseAdder, WhiteNoiseAdder
from aspire.source import ImageSource
from aspire.source.image import _ImageAccessor
from aspire.utils import (
    acorr,
    ainner,
    anorm,
    make_symmat,
    ratio_to_decibel,
    trange,
    uniform_random_angles,
    vecmat_to_volmat,
)
from aspire.utils.random import rand, randi, randn
from aspire.volume import AsymmetricVolume, Volume

logger = logging.getLogger(__name__)


class Simulation(ImageSource):
    """
    A `Simulation` represents a synthetic dataset of realistic cryo-EM images with corresponding
    `metadata`. The images are generated via projections of a supplied `Volume` object, `vols`, over
    orientations define by the Euler angles, `angles`. Various types of corruption, such as noise and
    CTF effects, can be added to the images by supplying a `Filter` object to the `noise_filter` or
    `unique_filters` arguments.
    """

    def __init__(
        self,
        L=None,
        n=1024,
        vols=None,
        states=None,
        unique_filters=None,
        filter_indices=None,
        offsets=None,
        amplitudes=None,
        dtype=None,
        C=2,
        angles=None,
        seed=0,
        memory=None,
        noise_adder=None,
    ):
        """
        A `Simulation` object that supplies images along with other parameters for image manipulation.

        :param L: Resolution of projection images (integer). Default is 8.
            If a `Volume` is provided `L` and `vols.resolution` must agree.
        :param n: The number of images to generate (integer).
        :param vols: A `Volume` object representing a stack of volumes.
            Default is generated with `volume.volume_synthesis.AsymmetricVolume`.
        :param states: A 1d array of n integers in the interval [0, C). The i'th integer indicates
            the volume stack index used to produce the i'th projection image. Default is a random set.
        :param unique_filters: A list of Filter objects to be applied to projection images.
        :param filter_indices: A 1d array of n integers indicating the `unique_filter` indices associated
            with each image. Default is a random set of filter indices, .ie the filters from `unique_filters`
            are randomly assigned to the stack of images.
        :param offsets: A n-by-2 array of coordinates to offset the images. Default is a normally
            distributed set of offsets. Set `offsets = 0` to disable offsets.
        :param amplitude: A 1d array of n amplitudes to scale the projection images. Default is
            a random set in the interval [2/3, 3/2]. Set `amplitude = 1` to disable amplitudes.
        :param dtype: dtype for the Simulation
        :param C: Number of Volumes used to generate projection images. The default is C=2.
            If a `Volume` object is provided this parameter is overridden and `self.C` = `self.vols.n_vols`.
        :param angles: A n-by-3 array of Euler angles for use in projection. Default is a random set.
        :param seed: Random seed.
        :param memory: str or None. The path of the base directory to use as a data store or None.
            If None is given, no caching is performed.
        :param noise_adder: Optionally append instance of `NoiseAdder`
            to generation pipeline.

        :return: A Simulation object.
        """

        self.seed = seed

        # If a Volume is not provided we default to the legacy Gaussian blob volume.
        # If a Simulation resolution or dtype is not provided, we default to L=8 and np.float32.
        if vols is None:
            self.vols = AsymmetricVolume(
                L=L or 8,
                C=C,
                seed=self.seed,
                dtype=dtype or np.float32,
            ).generate()
        else:
            if dtype is not None and vols.dtype != dtype:
                raise RuntimeError(
                    f"Explicit {self.__class__.__name__} dtype {dtype}"
                    f" does not match provided vols.dtype {vols.dtype}."
                    " Please change the Volume using `astype`"
                    " or the Simuation `dtype` argument."
                )
            # Assign the explicitly provided volume.
            self.vols = vols

        if not isinstance(self.vols, Volume):
            raise RuntimeError("`vols` should be a Volume instance or `None`.")

        # Infer the details from volume when possible.
        super().__init__(
            L=self.vols.resolution, n=n, dtype=self.vols.dtype, memory=memory
        )

        # If a user provides both `L` and `vols`, resolution should match.
        if L is not None and L != self.L:
            raise RuntimeError(
                f"Simulation must have the same resolution as the provided Volume."
                f" Provided vols.resolution = {self.vols.resolution} and L = {L}."
            )

        # We need to keep track of the original resolution we were initialized with,
        # to be able to generate projections of volumes later, when we are asked to supply images.
        self._original_L = self.L

        if offsets is None:
            offsets = self.L / 16 * randn(2, n, seed=seed).astype(dtype).T

        if amplitudes is None:
            min_, max_ = 2.0 / 3, 3.0 / 2
            amplitudes = min_ + rand(n, seed=seed).astype(dtype) * (max_ - min_)

        self.C = self.vols.n_vols

        if states is None:
            states = randi(self.C, n, seed=seed)
        self.states = states

        self._uniform_random_angles = False
        if angles is None:
            angles = uniform_random_angles(n, seed=seed, dtype=self.dtype)
            self._uniform_random_angles = True
        self.angles = angles

        if unique_filters is None:
            unique_filters = []
        self.unique_filters = unique_filters
        # sim_filters must be a deep copy so that it is not changed
        # when unique_filters is changed
        self.sim_filters = copy.deepcopy(unique_filters)

        # Create filter indices and fill the metadata based on unique filters
        if unique_filters:
            if filter_indices is None:
                filter_indices = randi(len(unique_filters), n, seed=seed) - 1
            self._populate_ctf_metadata(filter_indices)
            self.filter_indices = filter_indices
        else:
            self.filter_indices = np.zeros(n, dtype=int)

        self.offsets = offsets
        self.amplitudes = amplitudes

        if noise_adder is not None:
            logger.info(f"Appending {noise_adder} to generation pipeline")
            if not isinstance(noise_adder, NoiseAdder):
                raise RuntimeError("`noise_adder` should be subclass of NoiseAdder")
        self.noise_adder = noise_adder

        self._projections_accessor = _ImageAccessor(self._projections, self.n)
        self._clean_images_accessor = _ImageAccessor(self._clean_images, self.n)
        # For Simulation, signal can be computed directly from clean_images.
        self._signal_images = self.clean_images

    def _populate_ctf_metadata(self, filter_indices):
        # Since we are not reading from a starfile, we must construct
        # metadata based on the CTF filters by hand and set the values
        # for these columns
        #
        # class attributes of CTFFilter:
        CTFFilter_attributes = (
            "voltage",
            "defocus_u",
            "defocus_v",
            "defocus_ang",
            "Cs",
            "alpha",
        )
        # get the CTF parameters, if they exist, for each filter
        # and for each image (indexed by filter_indices)
        filter_values = np.zeros((len(filter_indices), len(CTFFilter_attributes)))
        for i, filt in enumerate(self.unique_filters):
            filter_values[filter_indices == i] = [
                getattr(filt, att, np.nan) for att in CTFFilter_attributes
            ]
        # set the corresponding Relion metadata values that we would expect
        # from a STAR file
        self.set_metadata(
            [
                "_rlnVoltage",
                "_rlnDefocusU",
                "_rlnDefocusV",
                "_rlnDefocusAngle",
                "_rlnSphericalAberration",
                "_rlnAmplitudeContrast",
            ],
            filter_values,
        )

    @property
    def projections(self):
        """
        Return projections of generated volumes, without applying filters/shifts/amplitudes/noise

        :param start: start index (0-indexed) of the start image to return
        :param num: Number of images to return. If None, *all* images are returned.
        :param indices: A numpy array of image indices. If specified, start and num are ignored.
        :return: An Image instance.
        """
        return self._projections_accessor

    def _projections(self, indices):
        """
        Accesses and returns projections as an `Image` instance. Called by self._projections_accessor
        """
        im = np.zeros(
            (len(indices), self._original_L, self._original_L), dtype=self.dtype
        )

        states = self.states[indices]
        unique_states = np.unique(states)
        for k in unique_states:
            idx_k = np.where(states == k)[0]
            rot = self.rotations[indices[idx_k], :, :]

            im_k = self.vols.project(vol_idx=k - 1, rot_matrices=rot)
            im[idx_k, :, :] = im_k.asnumpy()

        return Image(im)

    @property
    def clean_images(self):
        """
        Return projections with filters/shifts/amplitudes applied, but without noise.
        Subscriptable property.
        """
        return self._clean_images_accessor

    def _clean_images(self, indices):
        return self._images(indices, enable_noise=False)

    def _images(self, indices, enable_noise=True):
        """
        Returns particle images when accessed via the `ImageSource.images` property.
        :param indices: A 1-D NumPy array of integer indices.
        :param enable_noise: Only used internally, toggled off when `clean_images` requested.
        :return: An `Image` object.
        """
        # check for cached images first
        if self._cached_im is not None:
            logger.debug("Loading images from cache")
            return self.generation_pipeline.forward(
                Image(self._cached_im[indices, :, :]), indices
            )
        im = self.projections[indices]

        # apply original CTF distortion to image
        im = self._apply_sim_filters(im, indices)

        im = im.shift(self.offsets[indices, :])

        im *= self.amplitudes[indices].reshape(len(indices), 1, 1).astype(self.dtype)

        if enable_noise and self.noise_adder is not None:
            im = self.noise_adder.forward(im, indices=indices)

        # Finally, apply transforms to resulting Image
        return self.generation_pipeline.forward(im, indices)

    def _apply_sim_filters(self, im, indices):
        return self._apply_filters(
            im,
            self.sim_filters,
            self.filter_indices[indices],
        )

    def vol_coords(self, mean_vol=None, eig_vols=None):
        """
        Coordinates of simulation volumes in a given basis

        :param mean_vol: A mean volume in the form of a Volume Instance (default `mean_true`).
        :param eig_vols: A set of k volumes in a Volume instance (default `eigs`).
        :return:
        """
        if mean_vol is None:
            mean_vol = self.mean_true()
        if eig_vols is None:
            eig_vols = self.eigs()[0]

        assert isinstance(mean_vol, Volume)
        assert isinstance(eig_vols, Volume)

        vols = self.vols - mean_vol  # note, broadcast

        V = vols.to_vec()
        EV = eig_vols.to_vec()

        coords = EV @ V.T

        res = vols - Volume.from_vec(coords.T @ EV)
        res_norms = anorm(res.asnumpy(), (1, 2, 3))
        res_inners = mean_vol.to_vec() @ res.to_vec().T

        return coords.squeeze(), res_norms, res_inners

    def mean_true(self):
        return Volume(np.mean(self.vols, 0))

    def covar_true(self):
        eigs_true, lamdbas_true = self.eigs()
        eigs_true = eigs_true.T.to_vec()

        covar_true = eigs_true.T @ lamdbas_true @ eigs_true
        covar_true = vecmat_to_volmat(covar_true)

        return covar_true

    def eigs(self):
        """
        Eigendecomposition of volume covariance matrix of simulation
        :return: A 2-tuple:
            eigs_true: The eigenvectors of the volume covariance matrix in the form of Volume instance.
            lambdas_true: The eigenvalues of the covariance matrix in the form of a (C-1)-by-(C-1) diagonal matrix.
        """
        C = self.C
        vols_c = self.vols - self.mean_true()

        p = np.ones(C) / C
        # RCOPT, we may be able to do better here if we dig in.
        Q, R = qr(vols_c.to_vec().T, mode="economic")

        # Rank is at most C-1, so remove last vector
        Q = Q[:, :-1]
        R = R[:-1, :]

        w, v = eigh(make_symmat(R @ np.diag(p) @ R.T))
        eigs_true = Volume.from_vec((Q @ v).T)

        # Arrange in descending order (flip column order in eigenvector matrix)
        w = w[::-1]
        eigs_true = Volume(eigs_true[::-1])

        return eigs_true, np.diag(w)

    # TODO: Too many eval_* methods doing similar things - encapsulate somehow?

    def eval_mean(self, mean_est):
        mean_true = self.mean_true()
        return self.eval_vol(mean_true, mean_est)

    def eval_vol(self, vol_true, vol_est):
        norm_true = anorm(vol_true)

        err = anorm(vol_true - vol_est)
        rel_err = err / norm_true
        # RCOPT
        corr = acorr(vol_true.asnumpy(), vol_est.asnumpy())

        return {"err": err, "rel_err": rel_err, "corr": corr}

    def eval_covar(self, covar_est):
        covar_true = self.covar_true()
        return self.eval_volmat(covar_true, covar_est)

    def eval_volmat(self, volmat_true, volmat_est):
        """
        Evaluate volume matrix estimation accuracy

        :param volmat_true: The true volume matrices in the form of an L-by-L-by-L-by-L-by-L-by-L-by-K array.
        :param volmat_est: The estimated volume matrices in the same form.
        :return:
        """
        norm_true = anorm(volmat_true)

        err = anorm(volmat_true - volmat_est)
        rel_err = err / norm_true
        corr = acorr(volmat_true, volmat_est)

        return {"err": err, "rel_err": rel_err, "corr": corr}

    def eval_eigs(self, eigs_est, lambdas_est):
        """
        Evaluate covariance eigendecomposition accuracy

        :param eigs_est: The estimated volume eigenvectors in an L-by-L-by-L-by-K array.
        :param lambdas_est: The estimated eigenvalues in a K-by-K diagonal matrix (default `diag(ones(K, 1))`).
        :return:
        """
        eigs_true, lambdas_true = self.eigs()

        B = eigs_est.to_vec() @ eigs_true.to_vec().T
        norm_true = anorm(lambdas_true)
        norm_est = anorm(lambdas_est)

        inner = ainner(B @ lambdas_true, lambdas_est @ B)
        err = np.sqrt(norm_true**2 + norm_est**2 - 2 * inner)
        rel_err = err / norm_true
        corr = inner / (norm_true * norm_est)

        # TODO: Determine Principal Angles and return as a dict value

        return {"err": err, "rel_err": rel_err, "corr": corr}

    def eval_clustering(self, vol_idx):
        """
        Evaluate clustering estimation

        :param vol_idx: Indexes of the volumes determined (0-indexed)
        :return: Accuracy [0-1] in terms of proportion of correctly assigned labels
        """
        assert (
            len(vol_idx) == self.n
        ), f"Need {self.n} vol indexes to evaluate clustering"
        # Remember that `states` is 1-indexed while vol_idx is 0-indexed
        correctly_classified = np.sum(self.states - 1 == vol_idx)

        return correctly_classified / self.n

    def eval_coords(self, mean_vol, eig_vols, coords_est):
        """
        Evaluate coordinate estimation

        :param mean_vol: A mean volume in the form of a Volume instance.
        :param eig_vols: A set of eigenvolumes in an Volume instance.
        :param coords_est: The estimated coordinates in the affine space defined centered at `mean_vol` and spanned
            by `eig_vols`.
        :return:
        """
        assert isinstance(mean_vol, Volume)
        assert isinstance(eig_vols, Volume)
        coords_true, res_norms, res_inners = self.vol_coords(mean_vol, eig_vols)

        # 0-indexed states vector
        states = self.states - 1

        coords_true = coords_true[states]
        res_norms = res_norms[states]
        res_inners = res_inners[:, states]

        mean_eigs_inners = (mean_vol.to_vec() @ eig_vols.to_vec().T).item()
        coords_err = coords_true - coords_est

        err = np.hypot(res_norms, coords_err)

        mean_vol_norm2 = anorm(mean_vol) ** 2
        norm_true = np.sqrt(
            coords_true**2
            + mean_vol_norm2
            + 2 * res_inners
            + 2 * mean_eigs_inners * coords_true
        )
        norm_true = np.hypot(res_norms, norm_true)

        rel_err = err / norm_true
        inner = (
            mean_vol_norm2
            + mean_eigs_inners * (coords_true + coords_est)
            + coords_true * coords_est
            + res_inners
        )
        norm_est = np.sqrt(
            coords_est**2 + mean_vol_norm2 + 2 * mean_eigs_inners * coords_est
        )

        corr = inner / (norm_true * norm_est)

        return {"err": err, "rel_err": rel_err, "corr": corr}

    def estimate_snr(self, *args, **kwargs):
        """
        See ImageSource.estimate_snr() for documentation.
        """
        # For clean images return infinite SNR.
        # Note, relationship with CTF and other sim corruptions still isn't clear to me...
        if self.noise_adder is None:
            return np.inf

        # For SNR of Simulations, use the theoretical noise variance
        # known from the noise_adder instead of deriving from PSD.
        return super().estimate_snr(noise_power=self.noise_adder.noise_var)

    @classmethod
    def from_snr(
        cls,
        target_snr,
        *args,
        sample_n=None,
        support_radius=None,
        batch_size=512,
        signal_power_method="estimate_signal_mean_energy",
        **kwargs,
    ):
        """
        Generates a Simulation source with a WhiteNoiseAdder
        configured to produce a target signal to noise ratio.

        :param target_snr: Desired signal to noise ratio of
            the returned source.
        :param sample_n: Number of images used for estimate.
            Defaults to all images in source.
        :param support_radius: Pixel radius used for masking signal support.
            Default of None will compute inscribed circle, `self.L // 2`.
        :param batch_size: Images per batch, defaults 512.
        :param signal_power_method: Method used for computing signal energy.
           Defaults to mean via `estimate_signal_mean_energy`.
           Can use variance method via `estimate_signal_var`.
        :returns: Simulation source.
        """

        # Create a Simulation
        sim = cls(*args, **kwargs)

        # When a user supplies angles and requests `sample_n` subset of images
        #   randomize the rotations for them to avoid as much bias as possible.
        # The original rotations stored in `_rots` will be restored later.
        shuffle = (not sim._uniform_random_angles) and (sample_n is not None)
        if shuffle:
            # Store the original rotations.
            _rots = sim.rotations.copy()
            # Shuffle the Sim rotations
            sim.rotations = sim.rotations[np.random.permutation(sim.n)]

        # Assert NoiseAdder has not been provided
        if "noise_adder" in kwargs or sim.noise_adder is not None:
            raise RuntimeError(
                "Cannot provide 'noise_adder' when using {cls.__name__}.from_snr."
            )

        # Estimate the required noise variance
        signal_var = sim.estimate_signal_power(
            sample_n=sample_n,
            support_radius=support_radius,
            signal_power_method=signal_power_method,
        )
        noise_var = signal_var / target_snr

        # Assign the noise_adder
        sim.noise_adder = WhiteNoiseAdder(var=noise_var)
        logger.info(f"Appended {sim.noise_adder} to generation pipeline")

        # Restore the original rotations when previously shuffled.
        if shuffle:
            sim.rotations = _rots

        return sim

    def estimate_psnr(self, sample_n=None, batch_size=512, units=None):
        """
        Estimate Peak SNR as max(signal)^2 / MSE,
        where MSE is computed between `projections`
        and the resulting simulated `images`.
        PSNR is computed along the stack axis and an average
        value across the stack sample is returned.
        Note that PSNR is inherently a poor metric for identical images.

        :param sample_n: Number of images used for estimate.
            Defaults to all images in source.
        :param units: Optionally, convert from default ratio to log scale (`dB`).
        :returns: Estimated peak signal to noise ratio.
        """

        if sample_n is None:
            sample_n = self.n

        if sample_n > self.n:
            logger.warning(
                f"`estimate_psnr` sample_n > Source.n: {sample_n} > {self.n}."
                f" Accuracy may be impaired, settting sample_n=self.n={self.n}"
            )
            sample_n = self.n

        peaksq = np.empty(sample_n, dtype=self.dtype)
        mse = np.empty(sample_n, dtype=self.dtype)
        for start in trange(0, sample_n, batch_size):
            end = min(start + batch_size, sample_n)

            signal = self.projections[start:end].asnumpy()
            images = self.images[start:end].asnumpy()
            peaksq[start:end] = np.square(signal).max(axis=(-1, -2))  # max per image

            # Reshape and Transpose for Scikit metrics,
            # which expect a 2d (samples, outputs)
            mse[start:end] = mean_squared_error(
                signal.reshape(sample_n, -1).T,
                images.reshape(sample_n, -1).T,
                multioutput="raw_values",
            )

        psnr = peaksq / mse
        if units == "dB":
            psnr = ratio_to_decibel(psnr)
        elif units is not None:
            raise ValueError("Units should be `None` or `dB`.")

        # Return the mean psnr of the stack.
        return np.mean(psnr)
