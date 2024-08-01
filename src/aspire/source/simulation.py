import copy
import logging

import numpy as np
from scipy.linalg import eigh, qr
from sklearn.metrics import adjusted_rand_score

from aspire.image import Image
from aspire.noise import NoiseAdder
from aspire.source import ImageSource
from aspire.source.image import _ImageAccessor
from aspire.utils import (
    Rotation,
    acorr,
    ainner,
    anorm,
    make_symmat,
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
        symmetry_group=None,
        pixel_size=None,
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
        :param symmetry_group: A SymmetryGroup instance or string indicating symmetry of the molecule.
        :param pixel_size: Pixel size of the images in angstroms, default `None`.

        :return: A Simulation object.
        """

        self.seed = seed

        # If a Volume is not provided we default to the legacy Gaussian blob volume.
        # If a Simulation resolution or dtype is not provided, we default to L=8 and np.float32.
        if vols is None:
            self.vols = AsymmetricVolume(
                L=L or 8,
                C=C,
                pixel_size=pixel_size,
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

        if symmetry_group and (str(symmetry_group) != str(self.vols.symmetry_group)):
            logger.warning(
                f"Overriding {str(self.vols.symmetry_group)} symmetry group inherited "
                f"from `vols`, with user provided symmetry group: {str(symmetry_group)}."
            )
        symmetry_group = symmetry_group or self.vols.symmetry_group

        # Infer the details from volume when possible.
        super().__init__(
            L=self.vols.resolution,
            n=n,
            dtype=self.vols.dtype,
            memory=memory,
            symmetry_group=symmetry_group,
            pixel_size=self.vols.pixel_size,
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

        self.angles = self._init_angles(angles)

        if unique_filters is None:
            unique_filters = []
        self.unique_filters = unique_filters
        self._check_filter_pixel_size(unique_filters)
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

        self._projections_accessor = _ImageAccessor(self._projections, self.n)
        self._clean_images_accessor = _ImageAccessor(self._clean_images, self.n)

        # If a user prescribed NoiseAdder.from_snr(...),
        #   noise_adder will be a function returning a completed class.
        # Note the delayed eval may attempt to use self.*_accessors above.
        if noise_adder is not None:
            logger.info(f"Appending {noise_adder} to generation pipeline")
            # If we need to calculate signal_power from Simulation,
            # do so now and assign it to complete the Filter.
            if getattr(noise_adder, "requires_signal_power", False):
                noise_adder.signal_power = self.true_signal_power()

            # At this point we should have a fully baked NoiseAdder
            if not isinstance(noise_adder, NoiseAdder):
                raise RuntimeError("`noise_adder` should be subclass of NoiseAdder")

        self.noise_adder = noise_adder

        # Any further operations should not mutate this instance.
        self._mutable = False

    def _init_angles(self, angles):
        if angles is None:
            angles = uniform_random_angles(self.n, seed=self.seed, dtype=self.dtype)
        return angles

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

    def _check_filter_pixel_size(self, unique_filters):
        """
        Private method to ensure user provided filters match `Simulation` pixel size.

        When `Simulation.pixel_size` is not `None`, any
        `unique_filters` having a non-matching `pixel_size` attribute
        will raise.
        """

        # Skip when Simulation pixel_size is not explicitly provided.
        if self.pixel_size is None:
            return

        for f in unique_filters:
            f_pixel_size = getattr(f, "pixel_size", None)
            if f_pixel_size is not None and not np.isclose(
                f_pixel_size, self.pixel_size
            ):
                raise ValueError(
                    f"`Simulation.pixel_size` {self.pixel_size} does not match filter {f} pixel size {f_pixel_size}."
                    "Ensure provided `pixel_size` attributes match."
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
        Accesses and returns projections as an `Image` instance. Called by self._projections_accessor.
        """
        im = np.zeros(
            (len(indices), self._original_L, self._original_L), dtype=self.dtype
        )

        states = self.states[indices]
        unique_states = np.unique(states)
        for k in unique_states:
            idx_k = np.where(states == k)[0]
            rot = self.rotations[indices[idx_k], :, :]

            im_k = self.vols[k - 1].project(rot_matrices=rot)
            im[idx_k, :, :] = im_k.asnumpy()

        return Image(im, pixel_size=self.pixel_size)

    @property
    def clean_images(self):
        """
        Return projections with filters/shifts/amplitudes applied, but without noise.
        Subscriptable property.
        """
        return self._clean_images_accessor

    def _clean_images(self, indices):
        return self._images(indices, clean_images=True)

    def _images(self, indices, clean_images=False):
        """
        Returns particle images when accessed via the `ImageSource.images` property.

        :param indices: A 1-D NumPy array of integer indices.
        :param clean_images: Only used internally, toggled on when `clean_images` requested.
             Will skip accessing cache, skip noise, while applying CTF to projections.
        :return: An `Image` object.
        """
        # check for cached images first
        if not clean_images and self._cached_im is not None:
            logger.debug("Loading images from cache")
            return self.generation_pipeline.forward(self._cached_im[indices], indices)

        im = self.projections[indices]

        # apply original CTF distortion to image
        im = self._apply_sim_filters(im, indices)

        im = im.shift(self.offsets[indices, :])

        im *= self.amplitudes[indices].reshape(len(indices), 1, 1).astype(self.dtype)

        if not clean_images and self.noise_adder is not None:
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

    def true_signal_power(self, *args, **kwargs):
        """
        Estimate the signal power of `clean_images`.

        For usage, see `ImageSource.estimate_signal_power`.

        :returns: Estimated signal power of `clean_images`
        """

        # Note, in the future we can do something more clever here,
        # perhaps starting with the simulation Volume.  For now we
        # share code with ImageSource, so the method is at least
        # identical, up to using `clean_images`.  The method is also
        # the same as NoiseEstimator, up to ignoring the first moment
        # and a few optional parameters.
        kwargs["image_accessor"] = self.clean_images
        return self.estimate_signal_power(*args, **kwargs)

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
        eigs_true = Volume(eigs_true.asnumpy()[::-1])

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
        Evaluate clustering estimation using an adjusted Rand score.

        :param vol_idx: Indexes of the volumes determined (0-indexed)
        :return: Accuracy [-0.5, 1] in terms of proportion of correctly assigned labels.
            Identical clusters (up to a permutation) have a score of 1, random labeling
            will be close to 0, and discordant clusterings will be negative.
        """
        assert (
            len(vol_idx) == self.n
        ), f"Need {self.n} vol indexes to evaluate clustering"
        # Remember that `states` is 1-indexed while vol_idx is 0-indexed.

        return adjusted_rand_score(self.states - 1, vol_idx)

    def eval_coords(self, mean_vol, eig_vols, coords_est):
        """
        Evaluate coordinate estimation

        :param mean_vol: A mean volume in the form of a Volume instance.
        :param eig_vols: A set of eigenvolumes in an Volume instance.
        :param coords_est: The estimated coordinates in the affine space defined centered
            at `mean_vol` and spanned by `eig_vols`.
        :return: Dictionary containing error, relative error, and correlation for each set
            of estimated coordinates.
        """
        assert isinstance(mean_vol, Volume)
        assert isinstance(eig_vols, Volume)
        coords_true, res_norms, res_inners = self.vol_coords(mean_vol, eig_vols)

        # 0-indexed states vector
        states = self.states - 1

        coords_true = coords_true.T[states]
        res_norms = res_norms[states]
        res_inners = res_inners[:, states]

        if coords_est.ndim == 1:
            coords_est = coords_est[:, None]
            coords_true = coords_true[:, None]

        mean_eigs_inners = mean_vol.to_vec() @ eig_vols.to_vec().T
        coords_err = coords_true - coords_est

        K = coords_true.shape[-1]
        err = np.zeros((K, len(coords_true)))
        rel_err = np.zeros((K, len(coords_true)))
        corr = np.zeros((K, len(coords_true)))

        for k in range(K):
            err[k] = np.hypot(res_norms, coords_err[:, k])

            mean_vol_norm2 = anorm(mean_vol) ** 2
            norm_true = np.sqrt(
                coords_true[:, k] ** 2
                + mean_vol_norm2
                + 2 * res_inners
                + 2 * mean_eigs_inners[:, k] * coords_true[:, k]
            )
            norm_true = np.hypot(res_norms, norm_true)

            rel_err[k] = err[k] / norm_true
            inner = (
                mean_vol_norm2
                + mean_eigs_inners[:, k] * (coords_true[:, k] + coords_est[:, k])
                + coords_true[:, k] * coords_est[:, k]
                + res_inners
            )
            norm_est = np.sqrt(
                coords_est[:, k] ** 2
                + mean_vol_norm2
                + 2 * mean_eigs_inners[:, k] * coords_est[:, k]
            )

            corr[k] = inner / (norm_true * norm_est)

        return {"err": err, "rel_err": rel_err, "corr": corr}

    def true_snr(self, *args, **kwargs):
        """
        Compute SNR using `true_signal_power` and the noise power known to simulation.

        See Simulation.true_signal_power() for parameters.
        """
        # For clean images return infinite SNR.
        # Note, relationship with CTF and other sim corruptions still isn't clear to me...
        if self.noise_adder is None or self.noise_adder.noise_var == 0:
            return np.inf

        # For SNR of Simulations, use the theoretical noise variance
        # known from the noise_adder instead of deriving from PSD.
        noise_power = self.noise_adder.noise_var
        signal_power = self.true_signal_power(*args, **kwargs)
        return signal_power / noise_power


class _LegacySimulation(Simulation):
    """
    Legacy Simulation enforces the legacy grid convention for generating projection
    images.

    Note, that `angles`, and thus `rotations`, are altered upon initialization.
    To recover the rotations associated with the input angles use the staticmethod
    `rots_zyx_to_legacy_aspire()`.
    """

    def _init_angles(self, angles):
        angles = super()._init_angles(angles)

        # Convert to rotations.
        rots = Rotation.from_euler(angles).matrices

        # Transform rotations to replicate legacy grid convention.
        legacy_rots = Rotation(self.rots_zyx_to_legacy_aspire(rots))

        # Convert back to angles.
        return legacy_rots.angles.astype(self.dtype)

    @staticmethod
    def rots_zyx_to_legacy_aspire(rots):
        """
        Helper function to transform rotations to mimic original aspire python
        grid indexing. Now that we are enforcing "zyx" grid indexing across the
        code base, in particular for the rotated_grids used for volume projection,
        we must transform rotation matrices to allow for existing hardcoded tests
        to remain valid.

        Note, this transformation is it's own inverse.

        :param rots: n_rot x 3 x 3 array of rotation matrices.
        :return: Transformed rotations.
        """
        dtype = rots.dtype

        # Handle singletons
        og_shape = rots.shape
        if len(og_shape) == 2:
            rots = np.expand_dims(rots, axis=0)

        # Transform rots
        flip_xy = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=dtype)
        new_rots = rots[:, ::-1] @ flip_xy

        return new_rots.reshape(og_shape)
