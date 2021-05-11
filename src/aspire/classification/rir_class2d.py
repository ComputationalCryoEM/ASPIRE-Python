import logging

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
from aspire.numeric import ComplexPCA
from aspire.utils.random import rand

logger = logging.getLogger(__name__)


class RIRClass2D(Class2D):
    def __init__(
        self,
        src,
        pca_basis,
        fspca_components=400,
        alpha=1 / 3,
        sample_n=4000,
        bispectrum_componenents=300,
        n_nbor=100,
        bispectrum_freq_cutoff=None,
        large_pca_implementation="legacy",
        nn_implementation="legacy",
        bispectrum_implementation="devel",
        dtype=None,
    ):
        """
        Constructor of an object for classifying 2D images using
        Rotationally Invariant Representation (RIR) algorithm.

        Z. Zhao, Y. Shkolnisky, A. Singer, Rotationally Invariant Image Representation
        for Viewing Direction Classification in Cryo-EM. (2014)

        :param src: Source instance
        :param basis: (Fast) Fourier Bessel Basis instance
        :param fspca_components: Components (top eigvals) to keep from full FSCPA, default truncates to  400.
        :param sample_n: A number and associated method used to confuse your enemies.
        :param alpha: Amplitude Power Scale, default 1/3 (eq 20 from  RIIR paper).
        :param bispectrum_freq_cutoff: Truncate (zero) high k frequecies above (int) value, defaults off (None).
        :param dtype: optional dtype, otherwise taken from src.
        :return: RIRClass2D instance to be used to compute bispectrum-like rotationally invariant 2D classification.
        """
        super().__init__(src=src, dtype=dtype)

        self.pca_basis = pca_basis
        self.fb_basis = self.pca_basis.basis
        self.fspca_components = fspca_components
        self.sample_n = sample_n
        self.alpha = alpha
        self.bispectrum_componenents = bispectrum_componenents
        self.n_nbor = n_nbor
        self.bispectrum_freq_cutoff = bispectrum_freq_cutoff
        # Type checks
        if self.dtype != self.fb_basis.dtype:
            logger.warning(
                f"RIRClass2D basis.dtype {self.basis.dtype} does not match self.dtype {self.dtype}."
            )

        # Sanity Checks
        assert hasattr(self.pca_basis, "calculate_bispectrum")

        if self.src.n < self.bispectrum_componenents:
            raise RuntimeError(
                f"{self.src.n} Images too small for Bispectrum Componenents {self.bispectrum_componenents}."
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
        self._bispectrum = bispectrum_implementations[bispectrum_implementation]

        # For now, only run with FSPCA basis
        assert isinstance(self.pca_basis, FSPCABasis)

    def pca(self, M):
        """
        Any PCA implementation here should return both
        coef_b and coef_b_r that are (n_img, n_components).

        Where n_components is typically self.bispectrum_componenents.
        However, for small problems it may return n_components=n_img,
        since that would be the smallest dimension.

        To extend with an additional PCA like method,
        add as private method and list in large_pca_implementations.

        :param M: Array (n_img, m_features), typically complex.
        :returns: Tuple of arrays coef_b coef_b_r.
        """
        # _pca is assigned during initialization.
        return self._pca(M)

    def nn_classification(self, coef_b, coef_b_r):
        """
        Takes in features as pair of arrays (coef_b coef_b_r),
        each having shape (n_img, features)
        where features = min(self.bispectrum_componenents, n_img).

        Result is array (n_img, n_nbor) with entry i reprsenting
        index i into class input img array (src).

        To extend with an additonal Nearest Neighbor algo,
        add as a private method and list in nn_implementations.

        :returns: Array of indices representing classes.
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

    def classify(self):
        """
        Perform the 2D images classification.
        """

        # # Stage 1: Compute coef and reduce dimensionality.
        # Memioze/batch this later when result is working

        # Initial round of component truncation is before bispectrum.
        #  default of 400 components was taken from legacy code.
        # Instantiate a new compressed (truncated) basis.
        self.pca_basis = self.pca_basis.compress(self.fspca_components)

        # Expand into the compressed FSPCA space.
        fb_coef = self.fb_basis.evaluate_t(self.src.images(0, self.src.n))
        coef = self.pca_basis.expand(fb_coef)

        # Compute Bispectrum
        coef_b, coef_b_r = self.bispectrum(coef)

        # # Stage 2: Compute Nearest Neighbors
        logger.info("Begin Nearest Neighbors Search")
        classes = self.nn_classification(coef_b, coef_b_r)

        # # Stage 3: Align
        logger.info("Begin Rotational Alignment")
        return self.legacy_align(classes, coef)

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

        # any refl?
        logger.info(
            f"Count reflected: {np.sum(indices>=n_img)}"
            f" ({np.sum(indices>=n_img) / len(indices)}%)"
        )

        # # lets peek at the distribution of distances
        import matplotlib.pyplot as plt

        plt.hist(
            distances[:, 1:].flatten(), bins="auto"
        )  # zero index is self, distance 0
        plt.show()

        # I was planning to change the result of this function
        # but for now return classes as range 0 to 2*n_img..
        # #########################
        # # There are two sets of vectors each n_img long.
        # #   The second set represents reflected (gbw, unsure..)
        # #   When a reflected coef vector is a nearest neighbor,
        # #   we notate the original image index (indices modulus),
        # #   and notate we'll need the reflection (refl).
        # classes = indices % n_img
        # refl = np.array(indices // n_img, dtype=bool)
        # corr = distances
        # return classes, refl, corr

        return indices

    def output(self):
        """
        Return class averages.
        """
        pass

    def legacy_align(self, classes, coef):
        # translate some variables between this code and the legacy aspire aspire implementation (just trying to figure out of the old code ran...).
        freqs = self.pca_basis.complex_angular_indices
        coeff = coef.T
        n_im = self.src.n
        n_nbor = self.n_nbor

        # ## COPIED FROM LEGACY CODE:
        # del coeff_b, concat_coeff
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
        id_corr = np.argsort(-corr, axis=1)
        for i in range(n_im):
            corr[i] = corr[i, id_corr[i]]
            classes[i] = classes[i, id_corr[i]]
            rot[i] = rot[i, id_corr[i]]

        class_refl = (classes // n_im).astype(bool)
        classes[classes >= n_im] = classes[classes >= n_im] - n_im
        # # Why did they do this?
        #     rot[class_refl] = np.mod(rot[class_refl] + 180, 360) # ??
        rot *= np.pi / 180.0  # Convert to radians
        return classes, class_refl, rot, corr, 0

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

        num_batches = (
            n_im + batch_size - 1
        ) // batch_size  # int(np.ceil(float(n_im) / batch_size))

        classes = np.zeros((n_im, n_nbor), dtype=int)
        for i in range(num_batches):
            start = i * batch_size
            finish = min((i + 1) * batch_size, n_im)
            corr = np.real(
                # I dont understand what they were doing here yet.
                # I presume relying on dot being large for similar vectors.
                # But I don't get the conjugation etc.
                np.dot(np.conjugate(coeff_b[:, start:finish]).T, concat_coeff)
            )
            classes[start:finish] = np.argsort(-corr, axis=1)[:, 1 : n_nbor + 1]

        return classes

    def _legacy_pca(self, M):
        """
        PCA_y (y is I think for Yoel...).

        This is more or less the historic implementation ported
        to Python from MATLAB.
        """

        # ### The following was from legacy code. Be careful wrt order.
        M = M.T
        u, s, v = pca_y(M, self.bispectrum_componenents)

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
        # # Abandon using SK directly for now,
        # #   while it is really useful, it
        # #   expects real data.
        # #
        # # I tried the stupid things like
        # #   flattening,
        # #   running reals imags seperate,
        # #   running mags and phases seperately etc...
        # #   but the accuracy was too poor for me.
        # #   Should discuss this.
        # # So I subclassed sk PCA extended to complex numbers.
        pca = ComplexPCA(
            self.bispectrum_componenents,
            copy=False,  # careful, overwrites data matrix... we'll handle the copies.
            svd_solver="auto",  # use randomized (Halko) for larger problems
            random_state=123,
        )  # replace with ASPIRE repro seed later.
        coef_b = pca.fit_transform(M.copy())
        coef_b_r = pca.fit_transform(np.conjugate(M))

        # M is no longer needed, hint for it to get cleaned up.
        del M

        # I'm also not sure why this norm is needed...
        #  but it does work better with it.
        coef_b /= np.linalg.norm(coef_b, axis=1)[:, np.newaxis]
        coef_b_r /= np.linalg.norm(coef_b_r, axis=1)[:, np.newaxis]

        return coef_b, coef_b_r

    def _devel_bispectrum(self, coef):
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

        m = np.power(self.pca_basis.eigvals, self.alpha)
        m = m[
            self.pca_basis.complex_angular_indices != 0
        ]  # filter non_zero_freqs eq 18,19
        pm = m / np.sum(m)
        x = rand(len(m))
        m_mask = x < self.sample_n * pm

        M = None

        for i in tqdm(range(self.src.n)):
            B = self.pca_basis.calculate_bispectrum(
                coef_normed[i],
                filter_nonzero_freqs=True,
                freq_cutoff=self.bispectrum_freq_cutoff,
            )

            # ### Truncate Bispectrum (by sampling)
            # ### Note, where is this written down? (and is it even needed?)
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
            f"Computing Large PCA, returning {self.bispectrum_componenents} components."
        )
        # should add memory sanity check here... these can be crushingly large...
        coef_b, coef_b_r = self.pca(M)

        return coef_b, coef_b_r

    def _legacy_bispectrum(self, coef):
        """
        This code was ported to Python by an unkown author,
        and is the closest viable reference material.

        It is copied here to compare it a hot swappable manner while
        fresh code is developed for this class.
        """

        # Legacy code requires we unpack just a few things to call it.

        coef_b, coef_b_r = bispec_2drot_large(
            coeff=coef.T,
            freqs=self.pca_basis.complex_angular_indices,
            eigval=self.pca_basis.eigvals,
            alpha=self.alpha,
            sample_n=self.sample_n,
        )

        return coef_b.T, coef_b_r.T
