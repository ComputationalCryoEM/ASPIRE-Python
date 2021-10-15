import logging
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm, trange

from aspire.basis import FSPCABasis
from aspire.classification import Class2D
from aspire.classification.legacy_implementations import bispec_2drot_large, pca_y
from aspire.image import Image
from aspire.numeric import ComplexPCA
from aspire.source import ArrayImageSource
from aspire.utils.random import rand

logger = logging.getLogger(__name__)


class RIRClass2D(Class2D):
    def __init__(
        self,
        src,
        pca_basis=None,
        fspca_components=400,
        alpha=1 / 3,
        sample_n=4000,
        bispectrum_components=300,
        n_nbor=100,
        n_classes=50,
        bispectrum_freq_cutoff=None,
        large_pca_implementation="legacy",
        nn_implementation="legacy",
        bispectrum_implementation="legacy",
        alignment_implementation="bfr",
        alignment_opts=None,
        dtype=None,
        seed=None,
    ):
        """
        Constructor of an object for classifying 2D images using
        Rotationally Invariant Representation (RIR) algorithm.

        At a high level this consumes a Source instance `src`,
        and a FSPCA Basis `pca_basis`.

        Yield class averages by first performing `classify`,
        then performing `output`.

        Z. Zhao, Y. Shkolnisky, A. Singer, Rotationally Invariant Image Representation
        for Viewing Direction Classification in Cryo-EM. (2014)

        :param src: Source instance
        :param pca_basis: Optional FSPCA Basis instance
        :param fspca_components: Components (top eigvals) to keep from full FSCPA, default truncates to  400.
        :param alpha: Amplitude Power Scale, default 1/3 (eq 20 from  RIIR paper).
        :param sample_n: Threshold for random sampling of bispectrum coefs. Default 4000,
        high values such as 50000 reduce random sampling.
        :param n_nbor: Number of nearest neighbors to compute.
        :param n_classes: Number of class averages to return.
        :param bispectrum_freq_cutoff: Truncate (zero) high k frequecies above (int) value, defaults off (None).
        :param large_pca_implementation: See `pca`.
        :param nn_implementation: See `nn_classification`.
        :param bispectrum_implementation: See `bispectrum`.
        :param alignment_implementation: See `alignment`.
        :param alignment_opts: Optional implementation specific configuration options. See `alignment`.
        :param dtype: Optional dtype, otherwise taken from src.
        :param seed: Optional RNG seed to be passed to random methods, (example Random NN).
        :return: RIRClass2D instance to be used to compute bispectrum-like rotationally invariant 2D classification.
        """

        super().__init__(src=src, dtype=dtype)

        # For now, only run with FSPCA basis
        if pca_basis and not isinstance(pca_basis, FSPCABasis):
            raise NotImplementedError(
                "RIRClass2D has currently only been developed for pca_basis as a FSPCABasis."
            )

        self.pca_basis = pca_basis
        self.fspca_components = fspca_components
        self.sample_n = sample_n
        self.alpha = alpha
        self.bispectrum_components = bispectrum_components
        self.n_nbor = n_nbor
        self.n_classes = n_classes
        self.bispectrum_freq_cutoff = bispectrum_freq_cutoff
        self.seed = seed
        self.alignment_opts = alignment_opts

        if self.src.n < self.bispectrum_components:
            raise RuntimeError(
                f"{self.src.n} Images too small for Bispectrum Components {self.bispectrum_components}."
                "  Increase number of images or reduce components."
            )

        # Implementation Checks
        # # Do we have a sane Nearest Neighbor
        nn_implementations = {
            "legacy": self._legacy_nn_classification,
            "sklearn": self._sk_nn_classification,
        }
        if nn_implementation not in nn_implementations:
            raise ValueError(
                f"Provided nn_implementation={nn_implementation} not in {nn_implementations.keys()}"
            )
        self._nn_classification = nn_implementations[nn_implementation]

        # # Do we have a sane Large Dataset PCA
        large_pca_implementations = {
            "legacy": self._legacy_pca,
            "sklearn": self._sk_pca,
        }
        if large_pca_implementation not in large_pca_implementations:
            raise ValueError(
                f"Provided large_pca_implementation={large_pca_implementation} not in {large_pca_implementations.keys()}"
            )
        self._pca = large_pca_implementations[large_pca_implementation]

        # # Do we have a sane Bispectrum component
        bispectrum_implementations = {
            "legacy": self._legacy_bispectrum,
            "devel": self._devel_bispectrum,
        }
        if bispectrum_implementation not in bispectrum_implementations:
            raise ValueError(
                f"Provided bispectrum_implementation={bispectrum_implementation} not in {bispectrum_implementations.keys()}"
            )
        elif bispectrum_implementation == "legacy" and self._pca != self._legacy_pca:
            raise ValueError(
                '"legacy" bispectrum_implementation implies'
                ' large_pca_implementation="legacy".'
                " Check class configuration and retry."
            )
        self._bispectrum = bispectrum_implementations[bispectrum_implementation]

        alignment_implementations = {
            "bfr": self._bfr_align,
            "bfsr": self._bfsr_align,
        }
        if alignment_implementation not in alignment_implementations:
            raise ValueError(
                f"Provided alignment_implementation={alignment_implementation}"
                f" not in {alignment_implementations.keys()}."
            )
        self._alignment = alignment_implementations[alignment_implementation]

    def classify(self, diagnostics=False):
        """
        This is the high level method to perform the 2D images classification.

        The stages of this method are intentionally modular so they may be
        swapped for other implementations.

        :param diagnostics: Optionally plots distribution of distances
        """

        # # Stage 1: Compute coef and reduce dimensionality.
        # Memioze/batch this later when result is working
        # Initial round of component truncation is before bispectrum.
        #  default of 400 components was taken from legacy code.
        # Instantiate a new compressed (truncated) basis.
        if self.pca_basis is None:
            # self.pca_basis = self.pca_basis.compress(self.fspca_components)
            self.pca_basis = FSPCABasis(self.src, components=self.fspca_components)
        # For convenience, assign the fb_basis used in the pca_basis.
        self.fb_basis = self.pca_basis.basis

        # Get the expanded coefs in the compressed FSPCA space.
        self.fspca_coef = self.pca_basis.spca_coef

        # Compute Bispectrum
        coef_b, coef_b_r = self.bispectrum(self.fspca_coef)

        # # Stage 2: Compute Nearest Neighbors
        logger.info("Calculate Nearest Neighbors")
        classes, refl, distances = self.nn_classification(coef_b, coef_b_r)

        if diagnostics:
            # Lets peek at the distribution of distances
            # zero index is self, distance 0, ignored
            plt.hist(distances[:, 1:].flatten(), bins="auto")
            plt.show()

            # Report some information about reflections
            logger.info(f"Count reflected: {np.sum(refl)}" f" {100 * np.mean(refl) } %")

        # # Stage 3: Class Selection
        logger.info(f"Select {self.n_classes} Classes from Nearest Neighbors")
        # This is an area open to active research.
        # Currently we take a naive approach by selecting the
        # first n_classes assuming they are quasi random.
        classes = classes[: self.n_classes]

        # # Stage 4: Align
        logger.info(
            f"Begin Rotational Alignment of {classes.shape[0]} Classes using {self._alignment}."
        )
        return self.alignment(classes, refl, self.fspca_coef, self.alignment_opts)

    def pca(self, M):
        """
        Any PCA implementation here should return both
        coef_b and coef_b_r that are (n_img, n_components).

        `n_components` is typically self.bispectrum_components.
        However, for small problems it may return `n_components`=`n_img`,
        since that would be the smallest dimension.

        To extend class with an additional PCA like method,
        add as private method and list in `large_pca_implementations`.

        :param M: Array (n_img, m_features), typically complex.
        :returns: Tuple of arrays coef_b coef_b_r.
        """
        # _pca is assigned during initialization.
        return self._pca(M)

    def nn_classification(self, coef_b, coef_b_r):
        """
        Takes in features as pair of arrays (coef_b coef_b_r),
        each having shape (n_img, features)
        where features = min(self.bispectrum_components, n_img).

        Result is array (n_img, n_nbor) with entry `i` reprsenting
        index `i` into class input img array (src).

        To extend with an additonal Nearest Neighbor algo,
        add as a private method and list in nn_implementations.

        :param coef_b:
        :param coef_b_r:
        :returns:  Tuple of classes, refl, dists where
        classes is an integer array of indices representing image ids,
        refl is a bool array representing reflections (True is refl),
        and distances is an array of distances as returned by NN implementation.
        """
        # _nn_classification is assigned during initialization.
        return self._nn_classification(coef_b, coef_b_r)

    def bispectrum(self, coef):
        """
        All bispectrum implementations should consume a stack of fspca coef
        and return bispectrum coefficients.

        :param coef: complex steerable coefficients (eg. from FSPCABasis).
        :returns: tuple of arrays (coef_b, coef_b_r)
        """
        # _bispectrum is assigned during initialization.
        return self._bispectrum(coef)

    def _sk_nn_classification(self, coeff_b, coeff_b_r):
        # Before we get clever lets just use a generally accepted implementation.

        n_img = self.src.n

        # Third party tools generally expecting:
        #   slow axis as n_data, fast axis n_features.
        # Also most third party NN complain about complex...
        #   so we'll pretend we have 2*n_features of real values.
        # Don't worry about the copy because NearestNeighbors wants
        #   C-contiguous anyway... (it would copy internally otherwise)
        X = np.column_stack((coeff_b.real, coeff_b.imag))
        # We'll also want to consider the neighbors under reflection.
        #   These coefficients should be provided by coeff_b_r
        X_r = np.column_stack((coeff_b_r.real, coeff_b_r.imag))

        # We can compare both non-reflected and reflected representations as one large set by
        #   taking care later that we store refl=True where indices>=n_img
        X_both = np.concatenate((X, X_r))

        nbrs = NearestNeighbors(n_neighbors=self.n_nbor, algorithm="auto").fit(X_both)
        distances, indices = nbrs.kneighbors(X)

        # There were two sets of vectors each n_img long.
        #   The second set represented reflected.
        #   When a reflected coef vector is a nearest neighbor,
        #   we notate the original image index (indices modulus n_img),
        #   and notate we'll need the reflection (refl).
        classes = indices % n_img
        refl = np.array(indices // n_img, dtype=bool)

        return classes, refl, distances

        return indices

    def output(
        self,
        classes,
        classes_refl,
        rot,
        shifts=None,
        coefs=None,
    ):
        """
        Return class averages.

        :param classes: class indices (refering to src). (n_img, n_nbor)
        :param classes_refl: Bool representing whether to reflect image in `classes`
        :param rot: Array of in-plane rotation angles (Radians) of image in `classes`
        :param shifts: Optional array of shifts for image in `classes`.
        :coefs: Optional Fourier bessel coefs (avoids recomputing).
        :return: Stack of Synthetic Class Average images as Image instance.
        """

        logger.info(f"Select {self.n_classes} Classes from Nearest Neighbors")
        # generate indices for random sample (can do something smart with corr later).
        # For testing just take the first n_classes so it matches earlier plots for manual comparison
        # This is assumed to be reasonably random.
        selection = np.arange(self.n_classes)

        imgs = self.src.images(0, self.src.n)
        fb_avgs = np.empty((self.n_classes, self.fb_basis.count), dtype=self.src.dtype)

        for i in tqdm(range(self.n_classes)):
            j = selection[i]
            # Get the neighbors
            neighbors_ids = classes[j]

            # Get coefs in Fourier Bessel Basis if not provided as an argument.
            if coefs is None:
                neighbors_imgs = Image(imgs[neighbors_ids])
                if shifts is not None:
                    neighbors_imgs.shift(shifts[i])
                neighbors_coefs = self.fb_basis.evaluate_t(neighbors_imgs)
            else:
                neighbors_coefs = coefs[neighbors_ids]
                if shifts is not None:
                    neighbors_coefs = self.fb_basis.shift(neighbors_coefs, shifts[i])

            # Rotate in Fourier Bessel
            neighbors_coefs = self.fb_basis.rotate(
                neighbors_coefs, rot[j], classes_refl[j]
            )

            # Averaging in FB
            fb_avgs[i] = np.mean(neighbors_coefs, axis=0)

        # Now we convert the averaged images from FB to Cartesian.
        return ArrayImageSource(self.fb_basis.evaluate(fb_avgs))

    def alignment(self, classes, refl, coef, alignment_opts=None):
        """
        Any class averagiing alignment method should take in the following arguments and return the tuple described.

        The returned `classes` and `refl` should be same as the input.

        Returned `rot` is an (n_classes, n_nbor) array of angles which should represent the rotations needed to align images within that class. `rot` is measure in Radians.

        Returned `corr` is an (n_classes, n_nbor) array that should represent a correlation like measure between classified images and their base image (image index 0).

        Returned `shifts` is None or an (n_classes, n_nbor) array of 2D shifts which should represent the translation needed to best align the images within that class.


        Alignment implementations may admit specific conifguration options using an optional `alignment_opts` dictionary.

        :param classes: (n_classes, n_nbor) integer array of indices
        :param refl: (n_classes, n_nbor) bool array of reflections
        :param coef: (n_img, self.pca_basis.count) array of compressed basis coefficients.

        :returns: (classes, refl, rot, corr, shifts)
        """

        # _alignment is assigned during initialization.
        return self._alignment(classes, refl, coef, alignment_opts)

    def _bfsr_align(self, classes, refl, coef, alignment_opts=None):
        """
        This perfoms a Brute Force Shift and Rotational alignment.

        For each pair of x_shifts and y_shifts,
           Perform BFR

        Return the rotation and shift yielding the best results.
        """

        # Unpack any configuration options, or get defaults.
        if alignment_opts is None:
            alignment_opts = {}
        # Default shift search space of +- 1 in X and Y
        n_x_shifts = alignment_opts.get("n_x_shifts", 1)
        n_y_shifts = alignment_opts.get("n_y_shifts", 1)

        # Compute the shifts. Roll array so 0 is first.
        x_shifts = np.roll(np.arange(-n_x_shifts, n_x_shifts + 1), -n_x_shifts)
        y_shifts = np.roll(np.arange(-n_y_shifts, n_y_shifts + 1), -n_y_shifts)
        assert (x_shifts[0], y_shifts[0]) == (0, 0)

        # These arrays will incrementally store our best alignment.
        rots = np.empty(classes.shape, dtype=self.dtype)
        corr = np.ones(classes.shape, dtype=self.dtype) * -np.inf
        shifts = np.empty((*classes.shape, 2), dtype=int)

        # We want to maintain the original coefs for the base images,
        #  because we will mutate them with shifts in the loop.
        original_coef = coef[classes[:, 0], :]
        assert original_coef.shape == (self.n_classes, self.pca_basis.count)

        # Loop over shift search space, updating best result
        for x, y in product(x_shifts, y_shifts):
            shift = np.array([x, y], dtype=int)
            logger.info(f"Computing Rotational alignment after shift ({x},{y}).")

            # Shift the coef representing the first (base) entry in each class
            #   by the negation of the shift
            # Shifting one image is more efficient than shifting every neighbor
            coef[classes[:, 0], :] = self.pca_basis.shift(original_coef, -shift)

            _, _, _rots, _, _corr = self._bfr_align(classes, refl, coef, alignment_opts)

            # Each class-neighbor pair may have a best shift-rot from a different shift.
            # Test and update
            improved_indices = _corr > corr
            rots[improved_indices] = _rots[improved_indices]
            corr[improved_indices] = _corr[improved_indices]
            shifts[improved_indices] = shift

            # Restore unshifted base coefs
            coef[classes[:, 0], :] = original_coef

            if (x, y) == (0, 0):
                logger.info("Initial rotational alignment complete (shift (0,0))")
                assert np.sum(improved_indices) == np.size(
                    classes
                ), f"{np.sum(improved_indices)} =?= {np.size(classes)}"
            else:
                logger.info(
                    f"Shift ({x},{y}) complete. Improved {np.sum(improved_indices)} alignments."
                )

        return classes, refl, rots, shifts, corr

    def _bfr_align(self, classes, refl, coef, alignment_opts=None):
        """
        This perfoms a Brute Force Rotational alignment.

        For each class,
            constructs n_angles rotations of all class members,
            and then identifies angle yielding largest correlation(dot).

        For params, see `align`.

        Configurable `alignment_opts`:
        `n_angles` sets the number of brute force rotations to attempt.
        Defaults `n_angles=359`.
        """

        # Configure any alignment options, otherwise use defaults.
        if alignment_opts is None:
            alignment_opts = {}
        n_angles = alignment_opts.get("n_angles", 359)

        # Construct array of angles to brute force.
        test_angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)

        # Instantiate matrices for results
        rots = np.empty(classes.shape, dtype=self.dtype)
        corr = np.empty(classes.shape, dtype=self.dtype)
        results = np.empty((self.n_nbor, n_angles))

        for k in trange(self.n_classes):

            # Get the coefs for these neighbors
            nbr_coef = coef[classes[k]]

            for i, angle in enumerate(test_angles):
                # Rotate the set of neighbors by angle,
                rotated_nbrs = self.pca_basis.rotate(nbr_coef, angle, refl[k])

                # then store dot between class base image (0) and each nbor
                for j, nbor in enumerate(rotated_nbrs):
                    results[j, i] = np.dot(nbr_coef[0], nbor)

            # Now along each class, find the index of the angle reporting highest correlation
            angle_idx = np.argmax(results, axis=1)

            # Store that angle as our rotation for this image
            rots[k, :] = test_angles[angle_idx]

            # Also store the correlations for each neighbor
            for j in range(self.n_nbor):
                corr[k, j] = results[j, angle_idx[j]]

        # None is placeholder for shifts
        return classes, refl, rots, None, corr

    def _legacy_nn_classification(self, coeff_b, coeff_b_r, batch_size=2000):
        """
        Perform nearest neighbor classification and alignment.
        """

        # Note kept ordering from legacy code (n_features, n_img)
        coeff_b = coeff_b.T
        coeff_b_r = coeff_b_r.T

        n_im = self.src.n
        # Shouldn't have more neighbors than images
        n_nbor = self.n_nbor
        if n_nbor >= n_im:
            logger.warning(
                f"Requested {self.n_nbor} self.n_nbor, but only {n_im} images. Setting self.n_nbor={n_im-1}."
            )
            n_nbor = n_im - 1

        concat_coeff = np.concatenate((coeff_b, coeff_b_r), axis=1)

        num_batches = (n_im + batch_size - 1) // batch_size

        classes = np.zeros((n_im, n_nbor), dtype=int)
        distances = np.zeros((n_im, n_nbor), dtype=self.dtype)
        for i in range(num_batches):
            start = i * batch_size
            finish = min((i + 1) * batch_size, n_im)
            corr = np.real(
                np.dot(np.conjugate(coeff_b[:, start:finish]).T, concat_coeff)
            )
            # Note legacy did not include the original image?
            # classes[start:finish] = np.argsort(-corr, axis=1)[:, 1 : n_nbor + 1]
            # This now does include the original image
            # (Matches sklean implementation.)
            # Check with Joakim about preference.
            # I (GBW) think class[i] should have class[i][0] be the original image index.
            classes[start:finish] = np.argsort(-corr, axis=1)[:, :n_nbor]
            # Store the corr values for the n_nhors in this batch
            distances[start:finish] = np.take_along_axis(
                corr, classes[start:finish], axis=1
            )

        # There were two sets of vectors each n_img long.
        #   The second set represented reflected.
        #   When a reflected coef vector is a nearest neighbor,
        #   we notate the original image index (indices modulus n_img),
        #   and notate we'll need the reflection (refl).
        refl = np.array(classes // n_im, dtype=bool)
        classes %= n_im

        return classes, refl, distances

    def _legacy_pca(self, M):
        """
        This is more or less the historic implementation ported
        to Python from code calling MATLAB's `pca_y`.
        """

        # ### The following was from legacy code. Be careful wrt order.
        M = M.T
        u, s, v = pca_y(M, self.bispectrum_components)

        # Contruct coefficients
        coef_b = np.einsum("i, ij -> ij", s, np.conjugate(v))
        coef_b_r = np.conjugate(u.T).dot(np.conjugate(M))

        # Normalize
        coef_b /= np.linalg.norm(coef_b, axis=0)
        coef_b_r /= np.linalg.norm(coef_b_r, axis=0)

        # Transpose (this code was originally F order)
        coef_b = coef_b.T
        coef_b_r = coef_b_r.T

        return coef_b, coef_b_r

    def _sk_pca(self, M):
        # Avoiding SK directly for now,
        #   while it is really useful, it
        #   expects real data.
        #   We use an extension of SK that is hacked to admit complex.
        pca = ComplexPCA(
            self.bispectrum_components,
            copy=False,  # careful, overwrites data matrix... we'll handle the copies.
            svd_solver="auto",  # use randomized (Halko) for larger problems
            random_state=self.seed,
        )
        coef_b = pca.fit_transform(M.copy())
        coef_b_r = coef_b.conj()

        # I'm also not sure why this norm is needed...
        #  but it does work better with it.
        coef_b /= np.linalg.norm(coef_b, axis=1)[:, np.newaxis]
        coef_b_r /= np.linalg.norm(coef_b_r, axis=1)[:, np.newaxis]

        return coef_b, coef_b_r

    def _devel_bispectrum(self, coef):
        coef = self.pca_basis.to_complex(coef)
        # Take just positive frequencies, corresponds to complex indices.
        # Original implementation used norm of Complex values, here abs of Real.
        eigvals = np.abs(self.pca_basis.eigvals[self.pca_basis.signs_indices >= 0])

        # Legacy code included a sanity check:
        # non_zero_freqs = self.pca_basis.complex_angular_indices != 0
        # coef_norm = np.log(np.power(np.abs(coef[:,non_zero_freqs]), self.alpha)).all())
        # just stick to the paper (eq 20) for now , look at this more later.

        coef_normed = np.where(
            coef == 0, 0, coef / np.power(np.abs(coef), 1 - self.alpha)
        )  # should use an epsilon here...

        if not np.isfinite(coef_normed).all():
            raise ValueError("Coefs should be finite")

        # ## Compute and reduce Bispectrum
        m = np.power(eigvals, self.alpha)
        m = m[
            self.pca_basis.complex_angular_indices != 0
        ]  # filter non_zero_freqs eq 18,19
        pm = m / np.sum(m)
        x = rand(len(m))
        m_mask = x < self.sample_n * pm

        M = None

        for i in tqdm(range(self.src.n)):
            B = self.pca_basis.calculate_bispectrum(
                coef_normed[i, np.newaxis],
                filter_nonzero_freqs=True,
                freq_cutoff=self.bispectrum_freq_cutoff,
            )

            # ### Truncate Bispectrum (by sampling)
            # ### Note, where is this written down? (and is it even needed?)
            # It is only briefly mentioned in the paper and was more/less
            # soft disabled in the matlab code.
            B = B[m_mask][:, m_mask]
            logger.info(f"Truncating Bispectrum to {B.shape} ({np.size(B)}) coefs.")

            # B is symmetric, take lower triangle of first two axis.
            tril = np.tri(B.shape[0], dtype=bool)
            B = B[tril, :]
            logger.info(f"Symmetry reduced Bispectrum to {np.size(B)} coefs.")
            # B is sparse and should have same sparsity for any image up to underflows...
            B = B.ravel()[np.flatnonzero(B)]
            logger.info(f"Sparse (nnz) reduced Bispectrum to {np.size(B)} coefs.")

            # Legacy code had bispect flattened as CSR and some other hacks.
            #   For now, we'll compute it densely then take nonzeros.
            if M is None:
                # Instanstiate M with B's nnz size
                M = np.empty((self.src.n, B.shape[0]), dtype=coef.dtype)
            M[i] = B

        # Reduce dimensionality of Bispectrum sample with PCA
        logger.info(
            f"Computing Large PCA, returning {self.bispectrum_components} components."
        )
        # should add memory sanity check here... these can be crushingly large...
        coef_b, coef_b_r = self.pca(M)

        return coef_b, coef_b_r

    def _legacy_bispectrum(self, coef):
        """
        This code was ported to Python by an unkown author,
        and is the closest viable reference material.

        It is copied here to compare while
        fresh code is developed for this class.
        """

        # The legacy code expects the complex representation
        coef = self.pca_basis.to_complex(coef)
        complex_eigvals = self.pca_basis.to_complex(self.pca_basis.eigvals).reshape(
            self.pca_basis.complex_count
        )  # flatten

        coef_b, coef_b_r = bispec_2drot_large(
            coeff=coef.T,  # Note F style tranpose here and in return
            freqs=self.pca_basis.complex_angular_indices,
            eigval=complex_eigvals,
            alpha=self.alpha,
            sample_n=self.sample_n,
        )

        return coef_b.T, coef_b_r.T
