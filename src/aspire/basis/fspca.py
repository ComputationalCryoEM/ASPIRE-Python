import copy
import logging

import numpy as np
from tqdm import tqdm

from aspire.basis import SteerableBasis
from aspire.covariance import RotCov2D
from aspire.operators import BlkDiagMatrix
from aspire.utils import complex_type, make_symmat

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

    def __init__(self, source, basis, noise_var=None, adaptive_support=False):
        """
        Not sure if I sure actually inherit from Basis, the __init__ doesn't correspond well... later...
        :param noise_var: None estimates noise (default).
        0 forces "clean" treatment (no weighting).
        Other values assigned to noise_var.
        """

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
        self.count = self.basis.count
        self.complex_count = self.basis.complex_count
        self.angular_indices = self.basis.angular_indices
        self.radial_indices = self.basis.radial_indices
        self.complex_angular_indices = self.basis.complex_angular_indices
        self.complex_radial_indices = self.basis.complex_radial_indices
        self.noise_var = noise_var  # noise_var is handled during `build` call.

        # Support sizes
        self.fourier_support_size = 0.5  # Legacy c (sometimes called bandlimit)
        self.cartesian_support_size = (
            self.src.L // 2
        )  # Legacy r (sometimes called support_size)
        if adaptive_support:
            raise NotImplementedError("adaptive_support not implemented yet.")
        assert isinstance(
            self.cartesian_support_size, int
        ), "Cartesian support should be integer number of pixels."

        # self._build()  # hrmm how/when to kick off build, tricky

    def build(self, coef):
        # figure out a better name later, talked about using via batchcov but im pretty suspect...

        if self.noise_var is None:
            from aspire.noise import AnisotropicNoiseEstimator

            logger.info("Estimate the noise of images using anisotropic method.")
            self.noise_var = AnisotropicNoiseEstimator(self.src).estimate()
        logger.info(f"Setting noise_var={self.noise_var}")

        cov2d = RotCov2D(self.basis)
        covar_opt = {
            "shrinker": "frobenius_norm",
            "verbose": 0,
            "max_iter": 250,
            "iter_callback": [],
            "store_iterates": False,
            "rel_tolerance": 1e-12,
            "precision": "float64",
            "preconditioner": "identity",
        }
        self.mean_coef_est = cov2d.get_mean(coef)
        self.covar_coef_est = cov2d.get_covar(
            coef,
            mean_coeff=self.mean_coef_est,
            noise_var=self.noise_var,
            covar_est_opt=covar_opt,
        )

        # Create the arrays to be packed by _compute_spca
        self.eigvals = np.zeros(self.basis.count, dtype=self.dtype)

        self.eigvecs = BlkDiagMatrix.empty(2 * self.basis.ell_max + 1, dtype=self.dtype)

        self.spca_coef = np.zeros((self.src.n, self.basis.count), dtype=self.dtype)

        self._compute_spca(coef)

    def _compute_spca(self, coef):
        """
        Algorithm 2 from paper.
        """

        # number of images
        n = self.src.n

        # Noise Variance
        n_var = self.noise_var

        # Compute coefficient vector of mean image at zeroth component
        self.mean_coef_zero = np.mean(
            self.mean_coef_est[self.angular_indices == 0], axis=0
        )

        # Make the Data matrix (A_k)
        # This code is intentionally ripped off cov2d so that we can refactor both later.
        # Begin copy pasta.
        # Initialize a totally empty BlkDiagMatrix, build incrementally.
        A = BlkDiagMatrix.empty(0, dtype=coef.dtype)
        ell = 0
        mask = self.basis._indices["ells"] == ell
        A_k = coef[:, mask] - self.mean_coef_zero
        A.append(A_k)

        for ell in range(1, self.basis.ell_max + 1):  # ell is k, k is q, for now
            mask = self.basis._indices["ells"] == ell
            mask_pos = [
                mask[i] and (self.basis._indices["sgns"][i] == +1)
                for i in range(len(mask))
            ]
            mask_neg = [
                mask[i] and (self.basis._indices["sgns"][i] == -1)
                for i in range(len(mask))
            ]
            A_k = (coef[:, mask_pos] + coef[:, mask_neg]) / 2
           # A_k_refl = (coef[:, mask_pos] - coef[:, mask_neg]) / 2

            A.append(A_k)
            A.append(A_k)
            #A.append(A_k_refl)

        # # end copy pasta
        assert len(A) == len(self.covar_coef_est)

        # Foreach angular frequency (`k` in paper, `ells` in FB code)
        eigval_index = 0
        for angular_index, C_k in enumerate(self.covar_coef_est):

            # # Eigen/SVD,
            # XXX make_symmat? shouldn't covar already be sym?, what am I missing.
            # eigvals_k, eigvecs_k = np.linalg.eig(make_symmat(C_k))
            eigvals_k, eigvecs_k = np.linalg.eigh(C_k)

            # Determistically enforce eigen vector sign convention
            eigvecs_k = fix_signs(eigvecs_k)

            # Sort eigvals_k (gbw, are they not sorted already?!)
            sorted_indices = np.argsort(-eigvals_k)
            eigvals_k, eigvecs_k = (
                eigvals_k[sorted_indices],
                eigvecs_k[:, sorted_indices],
            )

            # These are the basis indices
            basis_inds = np.arange(eigval_index, eigval_index + len(eigvals_k))

            # Store the eigvals
            self.eigvals[basis_inds] = eigvals_k

            # Store the eigvecs, note this is a BlkDiagMatrix and is assigned incrementally.
            self.eigvecs[angular_index] = eigvecs_k

            # # Construct A_k, matrix of expansion coefficients a^i_k_q
            # #   for image i, angular index k, radial index q,
            # #   (around eq 31-33)
            # #   Rows radial indices, columns image i.
            # #
            # # Garrett would just call this a `data` matrix.
            # #
            # # We can extract this directly (up to transpose) from
            # #  complex_coef vector where ells == angular_index
            # #  then use the transpose so image stack becomes columns.

            # indices =( self.angular_indices == angular_index ) & (self.basis._indices["sgns"] == 1)

            # To compute new expansion coefficients using spca basis
            #   we combine the basis coefs using the eigen decomposition.
            # Note image stack slow moving axis, and requires transpose from other implementation.

            self.spca_coef[:, basis_inds] = np.einsum(
                "ji, kj -> ki", eigvecs_k, A[angular_index]
            )

            eigval_index += len(eigvals_k)

        # Sanity check we have same dimension of eigvals and basis coefs.
        if eigval_index != self.basis.count:
            raise RuntimeError(
                f"eigvals dimension {eigval_index} != basis coef count {self.basis.count}."
            )

        # Store a map of indices sorted by eigenvalue.
        # #   We don't resort then now because this would destroy the block diagonal structure.
        # #
        # # sorted_indices[i] is the ith most powerful eigendecomposition index
        # #
        # # We can pass a full or truncated slice of sorted_indices to any array indexed by
        # #  the coefs.
        self.sorted_indices = np.argsort(-np.abs(self.eigvals))

    def expand_from_image_basis(self, x):
        """
        Take an image in the standard coordinate basis and express as FSPCA coefs.

        Note each FSPCA coef corresponds to a linear combination Fourier Bessel
        basis vectors, described by an eigenvector in FSPCA.

        :param x:  The Image instance representing a stack of images in the
        standard 2D coordinate basis to be evaluated.
        :return: Stack of coefs in the FSPCABasis.
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
        :return: Stack of coefs in the FSPCABasis.
        """
        # apply linear combination defined by FSPCA (eigvecs)
        #  can try blk_diag here, but I think needs to be extended to non square...,
        #  or masked.
        # c_fspca = (self.eigvecs.apply(c_fb.T)).T
        eigvecs = self.eigvecs
        if isinstance(eigvecs, BlkDiagMatrix):
            eigvecs = eigvecs.dense()

        c_fspca = x @ eigvecs

        assert c_fspca.shape == (x.shape[0], self.count)

        return c_fspca

    def evaluate_to_image_basis(self, c):
        """
        Take FSPCA coefs and evaluate as image in the standard coordinate basis.

        :param c:  Stack of coefs in the FSPCABasis to be evaluated.
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

        # apply FSPCA eigenvector to coefs c, yields coefs in self.basis
        eigvecs = self.eigvecs
        if isinstance(eigvecs, BlkDiagMatrix):
            eigvecs = eigvecs.dense()

        return c @ eigvecs.T

    def compress(self, k):
        """
        Use the eigendecomposition to select the most powerful
        coefficients.

        Using those coefficients new indice mappings are constructed.

        :param k: Number of components (coef)
        :return: New FSPCABasis instance
        """

        if k >= self.count:
            logger.warning(
                f"Requested compression to {k} components,"
                f" but already {self.count}."
                "  Skipping compression."
            )
            return self

        # Create a deepcopy.
        result = copy.deepcopy(self)
        # result = FSPCABasis(self.src, self.basis)

        # Create compressed mapping
        result.compressed = True
        result.count = k
        compressed_indices = self.sorted_indices[: result.count]

        # NOTE, no longer blk_diag! ugh
        # Note can copy from self or result, should be same...
        result.eigvals = self.eigvals[compressed_indices]
        result.eigvecs = self.eigvecs.dense()[:, compressed_indices]
        result.spca_coef = self.spca_coef[:, compressed_indices]

        result.angular_indices = self.angular_indices[compressed_indices]
        result.radial_indices = self.radial_indices[compressed_indices]

        compressed_positive = self.basis._indices["sgns"][compressed_indices] == 1
        result.complex_angular_indices = self.angular_indices[
            compressed_indices & compressed_positive
        ]
        result.complex_radial_indices = self.radial_indices[
            compressed_indices & compressed_positive
        ]
        result.complex_count = np.sum(compressed_positive)

        return result

    def to_complex(self, coef):
        """
        Return complex valued representation of coefficients.
        This can be useful when comparing or implementing methods
        from literature.

        There is a corresponding method, to_real.

        :param coef: Coefficients from this basis.
        :return: Complex coefficent representation from this basis.
        """

        if coef.ndim == 1:
            coef = coef.reshape(1, -1)

        if coef.dtype not in (np.float64, np.float32):
            raise TypeError("coef provided to to_complex should be real.")

        # Pass through dtype precions, but check and warn if mismatched.
        dtype = complex_type(coef.dtype)
        if coef.dtype != self.dtype:
            logger.warning(
                f"coef dtype {coef.dtype} does not match precision of basis.dtype {self.dtype}, returning {dtype}."
            )

        # Return the same precision as coef
        imaginary = dtype(1j)

        ccoef = np.zeros((coef.shape[0], self.complex_count), dtype=dtype)

        ind = 0

        for ell in self.angular_indices:
            if ell==0:
                idx = np.arange(self.k_max[0], dtype=int)
                ccoef[:, idx] = coef[:, idx]
                ind_pos += np.size(idx)
            else:
                idx = ind + np.arange(self.k_max[ell], dtype=int)
                idx_pos = ind_pos + np.arange(self.k_max[ell], dtype=int)
                idx_neg = idx_pos + self.k_max[ell]

                ccoef[:, idx] = (coef[:, idx_pos] - imaginary * coef[:, idx_neg]) / 2.0
                ind_pos += 2 * self.k_max[ell]
                
            ind += np.size(idx)


        return ccoef

