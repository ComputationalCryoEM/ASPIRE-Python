import copy
import logging

import numpy as np
from tqdm import tqdm

from aspire.basis import SteerableBasis
from aspire.operators import BlkDiagMatrix
from aspire.utils import complex_type

logger = logging.getLogger(__name__)


# This function was shamelessly copied from class_averaging.py
#   I think ASPIRE has something similar, and if not, we should...
#   Move this later....
def fix_signs(u):
    """
    makes the matrix coloumn sign be by the biggest value
    :param u: matrix
    :return: matrix
    """
    b = np.argmax(np.absolute(u), axis=0)
    b = np.array([np.linalg.norm(u[b[k], k]) / u[b[k], k] for k in range(len(b))])
    u = u * b
    return u


class FSPCABasis(SteerableBasis):
    """
    A class for Fast Steerable Principal Component Analaysis basis.

    FSPCA is an extension to Fourier Bessel representations
    (provided asFBBasis2D/FFBBasis2D), which computes combinations of basis
    coefficients coresponding to the princicpal components of image(s)
    represented in the provided basis.

    The principal components are computed from eigen decomposition of the
    covariance matrix, and when evaluated into the real domain form
    the set of `eigenimages`.

    The algorithm is described in the publication:
    Z. Zhao, Y. Shkolnisky, A. Singer, Fast Steerable Principal Component Analysis,
    IEEE Transactions on Computational Imaging, 2 (1), pp. 1-12 (2016).​

    """

    def __init__(self, source, basis):
        """Not sure if I sure actually inherit from Basis, the __init__ doesn't correspond well... later..."""

        self.basis = basis
        self.src = source
        # check/warn dtypes
        self.dtype = self.src.dtype
        if self.basis.dtype != self.dtype:
            logger.warning(
                f"basis.dtype {self.basis.dtype} does not match"
                f" source {self.src.dtype}, using {self.dtype}."
            )

        self.compressed = False
        self.count = self.complex_count = self.basis.complex_count
        self.complex_angular_indices = self.basis.complex_angular_indices
        self.complex_radial_indices = self.basis.complex_radial_indices

        # self._build()  # hrmm how/when to kick off build, tricky
        # self.built = False

    def build(self, coef):
        # figure out a better name later, talked about using via batchcov but im pretty suspect...

        # We'll use the complex representation for the calculations
        complex_coef = self.basis.to_complex(coef)

        # Create the arrays to be packed by _compute_spca
        self.eigvals = np.zeros(
            self.basis.complex_count, dtype=complex_type(self.dtype)
        )  # should be real... either make real, or use as a check...
        self.eigvecs = BlkDiagMatrix.empty(
            self.basis.ell_max + 1, dtype=complex_type(self.dtype)
        )
        self.spca_coef = np.zeros(
            (self.src.n, self.basis.complex_count), dtype=complex_type(self.dtype)
        )

        noise_var = 0  # XXX, clean img only for now

        self._compute_spca(complex_coef, noise_var)

        # self.built = True

    def _compute_spca(self, complex_coef, noise_var):
        """
        Algorithm 2 from paper.
        """

        # number of images
        n = self.src.n

        # Compute coefficient vector of mean image at zeroth component
        mean_coef = np.mean(complex_coef[:, self.complex_angular_indices == 0], axis=0)

        # Foreach angular frequency (`k` in paper, `ells` in FB code)
        eigval_index = 0
        for angular_index in tqdm(range(0, self.basis.ell_max + 1)):

            # # Construct A_k, matrix of expansion coefficients a^i_k_q
            # #   for image i, angular index k, radial index q,
            # #   (around eq 31-33)
            # #   Rows radial indices, columns image i.
            # #
            # # We can extract this directly (up to transpose) from
            # #  complex_coef vector where ells == angular_index
            # #  then use the transpose so image stack becomes columns.

            indices = self.complex_angular_indices == angular_index
            A_k = complex_coef[:, indices].T

            lambda_var = self.basis.n_r / (2 * n)  # this isn't used in the clean regime

            # Zero angular freq is a special case
            if angular_index == 0:  # eq 33
                # de-mean
                # A_k = A_k - mean_coef.T[:, np.newaxis]
                A_k = (A_k.T - mean_coef).T
                # double count the zero case
                lambda_var *= 2

            # # Compute the covariance matrix representation C_k, eq 32
            # #   Note, I don't see relevance of special case 33...
            # #     C_k = np.real(A_k @ A_k.conj().T) / n
            # #   Einsum is performing the transpose, sometimes better performance.
            C_k = np.einsum("ij, kj -> ik", A_k, A_k.conj()).real / n

            # # Eigen/SVD,
            eigvals_k, eigvecs_k = np.linalg.eig(C_k)
            logger.debug(
                f"eigvals_k.shape {eigvals_k.shape} eigvecs_k.shape {eigvecs_k.shape}"
            )

            # Determistically enforce eigen vector sign convention
            eigvecs_k = fix_signs(eigvecs_k)

            # Sort eigvals_k (gbw, are they not sorted already?!)
            sorted_indices = np.argsort(-eigvals_k)
            eigvals_k, eigvecs_k = (
                eigvals_k[sorted_indices],
                eigvecs_k[:, sorted_indices],
            )

            if noise_var != 0:
                raise NotImplementedError("soon")
            else:
                # These are the complex basis indices
                basis_inds = np.arange(eigval_index, eigval_index + len(eigvals_k))

                # Store the eigvals
                self.eigvals[basis_inds] = eigvals_k

                # Store the eigvecs, note this is a BlkDiagMatrix and is assigned incrementally.
                self.eigvecs[angular_index] = eigvecs_k

                # # To compute new expansion coefficients using spca basis
                # #   we combine the basis coefs using the eigen decomposition.
                # Note image stack slow moving axis, and requires transpose from other implementation.
                self.spca_coef[:, basis_inds] = np.einsum(
                    "ji, jk -> ik", eigvecs_k, A_k
                ).T

                eigval_index += len(eigvals_k)

                # # Computing radial eigen vectors (functions) (eq 35) is not used? check

        # Sanity check we have same dimension of eigvals and (complex) basis coefs.
        if eigval_index != self.basis.complex_count:
            raise RuntimeError(
                f"eigvals dimension {eigval_index} != complex basis coef count {self.basis.complex_count}."
            )

        # Store a map of indices sorted by eigenvalue.
        # #   We don't resort then now because this would destroy the block diagonal structure.
        # #
        # # sorted_indices[i] is the ith most powerful eigendecomposition index
        # #
        # # We can pass a full or truncated slice of sorted_indices to any array indexed by
        # #  the complex coefs.
        self.sorted_indices = np.argsort(-np.abs(self.eigvals))

    def expand_from_image_basis(self, x):
        """
        Take an image in the standard coordinate basis and express as FSPCA coefs.

        Note each FSPCA coef corresponds to a linear combination Fourier Bessel
        basis vectors, described by an eigenvector in FSPCA.

        :param x:  The Image instance representing a stack of images in the
        standard 2D coordinate basis to be evaluated.
        :return: Stack of (complex) coefs in the FSPCABasis.
        """
        fb_coefs = self.basis.evaluate_t(x)
        return self.expand(fb_coefs)

    def expand(self, x):
        """
        Take a Fourier-Bessel coefs and express as FSPCA coefs.

        Note each FSPCA coef corresponds to a linear combination Fourier Bessel
        basis vectors, described by an eigenvector in FSPCA.

        :param x:  Coefs representing a stack in the
        Fourier Bessel basis.
        :return: Stack of (complex) coefs in the FSPCABasis.
        """
        c_fb = self.basis.to_complex(x)

        # apply linear combination defined by FSPCA (eigvecs)
        #  can try blk_diag here, but I think needs to be extended to non square...,
        #  or masked.
        # c_fspca = (self.eigvecs.apply(c_fb.T)).T
        eigvecs = self.eigvecs
        if isinstance(eigvecs, BlkDiagMatrix):
            eigvecs = eigvecs.dense()

        c_fspca = c_fb @ eigvecs

        assert c_fspca.shape == (x.shape[0], self.complex_count)

        return c_fspca

    def evaluate_to_image_basis(self, c):
        """
        Take FSPCA coefs and evaluate as image in the standard coordinate basis.

        :param c:  Stack of (complex) coefs in the FSPCABasis to be evaluated.
        :return: The Image instance representing a stack of images in the
        standard 2D coordinate basis..
        """
        c_fb = self.evaluate(c)

        return self.basis.evaluate(c_fb)

    def evaluate(self, c):
        """
        Take FSPCA coefs and evaluate to Fourier Bessel (self.basis) ceofs.

        :param c:  Stack of (complex) coefs in the FSPCABasis to be evaluated.
        :return: The (real) coefs representing a stack of images in self.basis
        """

        # apply FSPCA eigenvector to coefs c, yields complex_coefs in self.basis
        eigvecs = self.eigvecs
        if isinstance(eigvecs, BlkDiagMatrix):
            eigvecs = eigvecs.dense()

        cv = c @ eigvecs.T
        # convert to real reprsentation
        return self.basis.to_real(cv)

    def compress(self, k):
        """
        Use the eigendecomposition to select the most powerful
        coefficients.

        Using those coefficients new indice mappings are constructed.

        :param k: Number of components (coef)
        :return: New FSPCABasis instance
        """

        if k >= self.complex_count:
            logger.warning(
                f"Requested compression to {k} components,"
                f" but already {self.complex_count}."
                "  Skipping compression."
            )
            return self

        # Create a deepcopy.
        result = copy.deepcopy(self)
        # result = FSPCABasis(self.src, self.basis)

        # Create compressed mapping
        result.compressed = True
        result.count = result.complex_count = k
        compressed_indices = self.sorted_indices[:k]

        # NOTE, no longer blk_diag! ugh
        # Note can copy from self or result, should be same...
        result.eigvals = self.eigvals[compressed_indices]
        result.eigvecs = self.eigvecs.dense()[:, compressed_indices]
        result.spca_coef = self.spca_coef[:, compressed_indices]

        result.complex_angular_indices = self.complex_angular_indices[
            compressed_indices
        ]
        result.complex_radial_indices = self.complex_radial_indices[compressed_indices]

        return result
