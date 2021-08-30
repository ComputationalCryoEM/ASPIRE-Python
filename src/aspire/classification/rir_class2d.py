import logging

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from aspire.basis import FSPCABasis
from aspire.classification import Class2D
from aspire.classification.legacy_implementations import (
    bispec_2drot_large,
    pca_y,
    rot_align,
)
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
        # This is an area open to active research.
        # Currently we take naive approach documented later in `output`.
        # logger.info(f"Select {self.n_classes} Classes from Nearest Neighbors")

        # # Stage 4: Align
        logger.info(f"Begin Rotational Alignment of {classes.shape[0]} Classes")
        return self.legacy_align(classes, refl, self.fspca_coef)

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

    def output(self, classes, classes_refl, rot, coefs=None):
        """
        Return class averages.

        :param classes: class indices (refering to src). (n_img, n_nbor)
        :param classes_refl: Bool representing whether to reflect image in `classes`
        :param rot: Array represting rotation angle (Radians) of image in `classes`
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
                neighbors_coefs = self.fb_basis.evaluate_t(neighbors_imgs)
            else:
                neighbors_coefs = coefs[neighbors_ids]

            # Rotate in Fourier Bessel
            neighbors_coefs = self.fb_basis.rotate(
                neighbors_coefs, rot[j], classes_refl[j]
            )

            # Averaging in FB
            fb_avgs[i] = np.mean(neighbors_coefs, axis=0)

        # Now we convert the averaged images from FB to Cartestian.
        return ArrayImageSource(self.fb_basis.evaluate(fb_avgs))

    def legacy_align(self, classes, refl, coef):
        # Translate some variables between this code and the legacy aspire implementation
        freqs = self.pca_basis.complex_angular_indices
        coeff = self.pca_basis.to_complex(coef).T
        n_im = self.src.n
        n_nbor = self.n_nbor

        # ## COPIED FROM LEGACY CODE:
        max_freq = np.max(freqs)
        cell_coeff = []
        for i in range(max_freq + 1):
            cell_coeff.append(
                np.concatenate(
                    (coeff[freqs == i], np.conjugate(coeff[freqs == i])), axis=1
                )
            )

        # maybe pairs should also be transposed
        pairs = np.stack(
            (classes.flatten("F"), np.tile(np.arange(n_im), n_nbor)), axis=1
        )
        corr, rot = rot_align(max_freq, cell_coeff, pairs)

        rot = rot.reshape((n_im, n_nbor), order="F")
        classes = classes.reshape(
            (n_im, n_nbor), order="F"
        )  # this should already be in that shape
        corr = corr.reshape((n_im, n_nbor), order="F")
        # Note that the sorting here for alignment is wrt correlation,
        #  whereas previously in the NN calculation sorting is by bispectrum distance.
        id_corr = np.argsort(-corr, axis=1)
        for i in range(n_im):
            corr[i] = corr[i, id_corr[i]]
            classes[i] = classes[i, id_corr[i]]
            rot[i] = rot[i, id_corr[i]]

        # Check Reflections usually imply rotation by 180, but this seems to yield worse results.
        # rot[class_refl] = np.mod(rot[class_refl] + 180, 360)
        rot *= np.pi / 180.0  # Convert to radians
        return classes, refl, rot, corr

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
