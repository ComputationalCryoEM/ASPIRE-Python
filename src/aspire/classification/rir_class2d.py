import logging

import numpy as np
from tqdm import tqdm

from aspire.classification import Class2D
from aspire.covariance import RotCov2D

logger = logging.getLogger(__name__)


class RIRClass2D(Class2D):
    def __init__(self, src, basis, eigvals=None, dtype=None):
        """
        Constructor of an object for classifying 2D images using
        Rotationally Invariant Representation (RIP) algorithm.

        :param src: Source instance
        :param basis: (Fast) Fourier Bessel Basis instance
        :eigavals: optional stack of eigvals of covariance matrices corresponding to stack of images in src.
        :param dtype: optional dtype, otherwise taken from src.
        :return: RIRClass2D instance to be used to compute bispectrum-like rotationally invariant 2D classification.
        """
        super().__init__(src=src, dtype=dtype)
        self.basis = basis
        if self.dtype != self.basis.dtype:
            logger.warning(
                f"RIRClass2D basis.dtype {self.basis.dtype} does not match self.dtype {self.dtype}."
            )

        # Memioze/batch this later when result is working
        self.coef = self.basis.evaluate_t(
            src.images(0, self.src.n)
        )  # basis coefficients

        self.freqs = self.basis.get_freqs()  # basis frequencies in cartesian

        # Get k
        self.angular_indices = (
            self.basis.get_angular_indices()
        )  # Map coef index to angular indices
        self.uniq_angular_indices = np.unique(self.angular_indices)

        # Get q
        self.radial_indices = self.basis.get_radial_indices()
        self.uniq_radial_indices = np.unique(self.radial_indices)

        logger.info(
            f"Unique Angular Indices {len(self.uniq_angular_indices)}\n"
            f"Unique Radial Indices {len(self.uniq_radial_indices)}"
        )

        # TODO noise_var, for now use clean...
        if eigvals is None:
            # Get the covar, returned as block diagonal matrix.
            blk_diag_cov = RotCov2D(self.basis).get_covar(coeffs=self.coef, noise_var=0)
            eigvals = blk_diag_cov.eigvals()

        assert len(eigvals) == len(
            self.coef[-1]
        ), f"{eigvals.shape} != {self.coef.shape}"
        self.eigvals = eigvals

    def calc_bispec(self, image_index, alpha=1 / 3):
        """
        Calculate bispectrum from the 2D image source provided at instance instantiation.
        """
        logger.info(f"Calculating bispectrum using {self.basis}")

        coef = self.coef[image_index]
        coef = np.log(np.power(np.absolute(coef), alpha))  # EQ 20
        assert np.all(np.isfinite(coef)), "Norm'd Coefficients should be finite."

        # yikes, B is large now....
        B = np.zeros(
            (self.basis.count, self.basis.count, self.uniq_radial_indices.shape[0]),
            dtype=self.dtype,
        )
        B_map = dict()  # maps (k1, q1, k2, q2) to B (c1, c2)

        max_freq = np.max(self.freqs)

        for ind1, coef1 in tqdm(enumerate(coef), total=coef.shape[0]):
            k1 = self.angular_indices[ind1]
            # q1 = self.radial_indices[ind1]

            for ind2, coef2 in enumerate(coef):
                k2 = self.angular_indices[ind2]
                # q2 = self.radial_indices[ind1]
                k3 = k1 + k2

                # B_map[(k1, q1, k2, q2)] = (ind1, ind2)
                intermodulated_coef_inds = np.where(self.angular_indices == k3)

                for coef3_ind in intermodulated_coef_inds:
                    q3 = self.radial_indices[coef3_ind]
                    coef3 = coef[coef3_ind]

                    B[ind1, ind2, q3] = coef1 * coef2 * coef3

        return B  # , Bmap

    def get_rand_pca(self, ind, alpha=1 / 3, n=4000):
        """
        Calculate PCA using randomized algorithm rotationally.
        """

        # Get the full bispectrum
        B = self.calc_bispec(ind)

        # Compute some normalized sequence M (corresponding to eigenvalue strength of coefs)
        #   That is eigvalue index i corresponds to basis coef i.
        M = np.power(self.eigvals, alpha)
        M /= M.sum()

        # Compute a vector of random values
        X = np.random.rand(len(M))  # would we just reuse this rand vector?
        # Compute a truncated representantive for M
        M_indices = X < n * M
        print("M.shape", M.shape)
        print(M_indices.shape)
        print(M_indices)

        # From the truncated representation for M we can select a truncated bispectrum.
        print("B.shape", B.shape)
        truncated_B = B[M_indices][:, M_indices]

        logger.info(
            f"Truncated bispectrum to {truncated_B.shape} with sparsity "
            f"{np.count_nonzero(truncated_B)/np.size(truncated_B) * 100.}%"
        )
        # Along with the truncated eigenvalues
        truncated_M = np.exp(truncated_B)

        # Now take the svd of the truncated bispectrum to yield princple components
        # logger.info("Computing SVD of truncated bispectrum")
        return truncated_B

    def nn_classification(self):
        """
        Perform nearest neighbor classification and alignment.
        """
        pass

    def classify(self):
        """
        Perform the 2D images classification.
        """
        pass

    def output(self):
        """
        Return class averages.
        """
        pass
