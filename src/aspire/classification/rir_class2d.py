import logging

import numpy as np
from scipy.linalg import qr
from tqdm import tqdm

from aspire.basis import FSPCABasis
from aspire.classification import Class2D
from aspire.covariance import RotCov2D
from aspire.utils import complex_type
from aspire.utils.random import rand

logger = logging.getLogger(__name__)

import matplotlib.pyplot as plt


class RIRClass2D(Class2D):
    def __init__(self, src, pca_basis, fspca_components=400, alpha=1/3,
                 rank_approx=4000,
                 bispectrum_componenents=300, dtype=None):
        """
        Constructor of an object for classifying 2D images using
        Rotationally Invariant Representation (RIR) algorithm.

        Z. Zhao, Y. Shkolnisky, A. Singer, Rotationally Invariant Image Representation
        for Viewing Direction Classification in Cryo-EM. (2014)

        :param src: Source instance
        :param basis: (Fast) Fourier Bessel Basis instance
        :param fspca_components: Components (top eigvals) to keep from full FSCPA, default truncates to  400.
        :param rank_approx: A number and associated method used to confuse your enemies.
        :param alpha: Amplitude Power Scale, default 1/3 (eq 20 from  RIIR paper).
        :param dtype: optional dtype, otherwise taken from src.
        :return: RIRClass2D instance to be used to compute bispectrum-like rotationally invariant 2D classification.
        """
        super().__init__(src=src, dtype=dtype)

        self.pca_basis = pca_basis
        self.fb_basis = self.pca_basis.basis
        self.fspca_components = fspca_components
        self.rank_approx = rank_approx
        self.alpha = alpha
        self.bispectrum_componenents = bispectrum_componenents

        # Type checks
        if self.dtype != self.fb_basis.dtype:
            logger.warning(
                f"RIRClass2D basis.dtype {self.basis.dtype} does not match self.dtype {self.dtype}."
            )

        # Sanity Checks
        assert hasattr(self.pca_basis, "calculate_bispectrum")

        # For now, only run with FSPCA
        assert isinstance(self.pca_basis, FSPCABasis)

    def classify(self):
        """
        Perform the 2D images classification.
        """

        ## Stage 1: Compute coef and reduce dimensionality.
        # Memioze/batch this later when result is working

        # Initial round of component truncation is before bispectrum.
        #  default of 400 components taken from legacy code.
        #  Take minumum here in case we have less than k coefs already.
        if self.fb_basis.complex_count > self.fspca_components:
            # Instantiate a new truncated (compressed) basis.
            self.pca_basis = self.pca_basis.truncate(self.fspca_components)

        # Expand into the compressed FSPCA space.
        fb_coef = self.fb_basis.evaluate_t(self.src.images(0, self.src.n))
        coef = self.pca_basis.expand(fb_coef)
        ## should be equiv, make a ut
        # coef = self.pca_basis.expand_from_image_basis(self.src.images(0, self.src.n))

        # Legacy code included a sanity check:
        non_zero_freqs = self.pca_basis.complex_angular_indices != 0
        #coef_norm = np.log(np.power(np.abs(coef[:,non_zero_freqs]), self.alpha)).all())
        # just stick to the paper (eq 20) for now , look at this more later.
        coef_normed = np.where(coef == 0, 0,
                             coef / np.power(np.abs(coef), 1-self.alpha))
        
        if not np.isfinite(coef_normed).all():
            raise ValueError("Coefs should be finite")

        ### Compute and reduce Bispectrum
        num_radial_freqs = len(np.unique(self.pca_basis.complex_radial_indices))
        coef_b = np.empty((self.src.n, num_radial_freqs, num_radial_freqs), dtype=coef.dtype)
        coef_b_r = np.empty_like(coef_b)
        
        for i in range(self.src.n):
            B = self.pca_basis.calculate_bispectrum(coef_normed[i])

            #### Truncate Bispectrum (by sampling)
            #### Note, where is this written down? Check if this is the rank_approx method mentioned in paper...
            M = np.power(self.pca_basis.eigvals, self.alpha)
            pM = M/ np.sum(M)            
            X = rand(len(M))
            M_mask = X < self.rank_approx * pM
            B = B[M_mask][:, M_mask]
            logger.info(f"Truncating Bispectrum to {B.shape} coefs.")            

            ### Reduce dimensionality of Bispectrum sample with PCA 
            # Legacy code had bispect with shape number_feasible_k3, non_zero_unique_radial_indices.
            M = B.reshape(B.shape[0] * B.shape[1], B.shape[2])

            logger.info(f"Computing PCA, returning {self.bispectrum_componenents} components.")
            u, s, v = self.pca_y(M, self.bispectrum_componenents)
            # ## # Check it looks something like a spectrum.
            # import matplotlib.pyplot as plt
            # plt.semilogy(s)
            # plt.show()
            ## Contruct coefficients
            coef_b[i] = np.einsum('i, ij -> ij', s, np.conjugate(v))
            coef_b_r[i] = np.conjugate(u.T).dot(np.conjugate(M))

            # normalize
            coef_b[i] /= np.linalg.norm(coef_b[i], axis=0)
            coef_b_r[i] /= np.linalg.norm(coef_b_r[i], axis=0)
            print(coef_b[i])
            import matplotlib.pyplot as plt
            plt.imshow(np.abs(coef_b[i]))
            plt.show()
            plt.imshow(np.abs(coef_b_r[i]))
            plt.show()
            

        ## Stage 2: Compute Nearest Neighbors
        classes = self.nn_classification(coef_b, coef_b_r)
        print(classes)

        ## Stage 3: Align

    def nn_classification(self, coeff_b, coeff_b_r, batch_size=2000):
        """
        Perform nearest neighbor classification and alignment.
        """
        # revisit

        concat_coeff = np.concatenate(coeff_b, coeff_b_r)
        
        num_batches = int(np.ceil(1.0 * n_im / batch_size))
        classes = np.zeros((n_im, n_nbor), dtype='int')
        for i in range(num_batches):
            start = i * batch_size
            finish = min((i + 1) * batch_size, n_im)
            corr = np.real(np.dot(np.conjugate(coeff_b[:, start: finish]).T, concat_coeff))
            classes[start: finish] = np.argsort(-corr, axis=1)[:, 1: n_nbor + 1]

        return classes

    def output(self):
        """
        Return class averages.
        """
        pass

    def pca_y(self, x, k, num_iters=2):
        """
        PCA using QR factorization. 
        
        See:

        An algorithm for the principal component analysis of large data sets.
        Halko, Martinsson, Shkolnisky, Tygert , SIAM 2011.
        
        :param x: Data matrix
        :param k: Number of estimated Principal Components.
        :param num_iters: Number of dot product applications.
        :return: (left Singular Vectors, Singular Values, right Singular Vectors)
        """

        # TODO, move this out of this class, its a general method...
        # Note I should come back, read this paper, and understand this better, looks useful.

        m, n = x.shape

        def operator(mat):
            return x.dot(mat)

        def operator_transpose(mat):
            return np.conj(x.T).dot(mat)

        flag = False
        if m < n:
            flag = True
            operator_transpose, operator = operator, operator_transpose
            m, n = n, m

        ones = np.ones((n, k + 2))
        if x.dtype == np.dtype('complex'):
            h = operator((2 * np.random.random((k + 2, n)).T - ones) + 1j * (2 * np.random.random((k + 2, n)).T - ones))
        else:
            h = operator(2 * np.random.random((k + 2, n)).T - ones)

        f = [h]

        for i in range(num_iters):
            h = operator_transpose(h)
            h = operator(h)
            f.append(h)

        f = np.concatenate(f, axis=1)
        # f has e-16 error, q has e-13
        q, _, _ = qr(f, mode='economic', pivoting=True)
        b = np.conj(operator_transpose(q)).T
        u, s, v = np.linalg.svd(b, full_matrices=False)
        # not sure how to fix the signs but it seems like I dont need to
        # TODO use fix_svd, here and matlab
        # u, v = fix_svd(u, v)

        v = v.conj()
        u = np.dot(q, u)

        u = u[:, :k]
        v = v[:k]
        s = s[:k]

        if flag:
            u, v = v.T, u.T

        return u, s, v
