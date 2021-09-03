import copy
import logging
from collections import OrderedDict

import numpy as np

from aspire.basis import FFBBasis2D, SteerableBasis2D
from aspire.covariance import RotCov2D
from aspire.operators import BlkDiagMatrix
from aspire.utils import complex_type, fix_signs, real_type

logger = logging.getLogger(__name__)


class FSPCABasis(SteerableBasis2D):
    """
    A class for Fast Steerable Principal Component Analaysis basis.

    FSPCA is an extension to Fourier Bessel representations
    (provided asF BBasis2D/FFBBasis2D), which computes combinations of basis
    coefficients coresponding to the princicpal components of image(s)
    represented in the provided basis.

    The principal components are computed from eigen decomposition of the
    covariance matrix, and when evaluated into the real domain and reshaped form
    the set of `eigenimages`.

    The algorithm is described in the publication:
    Z. Zhao, Y. Shkolnisky, A. Singer, Fast Steerable Principal Component Analysis,
    IEEE Transactions on Computational Imaging, 2 (1), pp. 1-12 (2016).

    """

    def __init__(self, src, basis=None, noise_var=None, components=400):
        """

        :param src: Source instance
        :param basis: Optional Fourier Bessel Basis (usually FFBBasis2D)
        :param noise_var: None estimates noise (default).
        0 forces "clean" treatment (no weighting).
        Other values assigned to noise_var.
        """

        self.src = src

        # Automatically generate basis if needed.
        if basis is None:
            basis = FFBBasis2D((self.src.L,) * 2, dtype=self.src.dtype)
        self.basis = basis

        # Components are used for `compress` during `build`.
        self.components = components

        # check/warn dtypes
        self.dtype = self.src.dtype
        if self.basis.dtype != self.dtype:
            logger.warning(
                f"basis.dtype {self.basis.dtype} does not match"
                f" source {self.src.dtype}, using {self.dtype}."
            )

        self.count = self.basis.count
        self.complex_count = self.basis.complex_count
        self.angular_indices = self.basis.angular_indices
        self.radial_indices = self.basis.radial_indices
        self.signs_indices = self.basis._indices["sgns"]
        self.complex_angular_indices = self.basis.complex_angular_indices
        self.complex_radial_indices = self.basis.complex_radial_indices

        self.complex_indices_map = self._get_complex_indices_map()
        assert (
            len(self.complex_indices_map) == self.complex_count
        ), f"{len(self.complex_indices_map)} != {self.complex_count}"

        self.noise_var = noise_var  # noise_var is handled during `build` call.

        self.build()

    def _get_complex_indices_map(self):
        """
        Private method for building an ordered dictionary mapping
        every complex coefficients (ell, q) pair
        with the real coefficients' flattened array index.

        The ordering stores the first occurance of a complex
        coefficient, which is used in compression to select the
        first k complex components.

        Once we know the first k complex components, this mapping
        further provides the index of both +/- real coefficients.

        This is subtly different than taking the top 2*k real
        components' coefficients and converting to complex,
        which often does not yield k complex components.
        """

        complex_indices_map = OrderedDict()
        for i in range(self.count):
            ell = self.angular_indices[i]
            q = self.radial_indices[i]
            sgn = self.signs_indices[i]

            complex_indices_map.setdefault((ell, q), [None, None])
            if sgn == 1:
                complex_indices_map[(ell, q)][0] = i
            elif sgn == -1:
                complex_indices_map[(ell, q)][1] = i
            else:
                raise ValueError("sgn should be +-1")

        return complex_indices_map

    def build(self):
        """
        Computes the FSPCA basis.

        This may take some time for large image stacks.
        """

        coef = self.basis.evaluate_t(self.src.images(0, self.src.n))

        if self.noise_var is None:
            from aspire.noise import WhiteNoiseEstimator

            logger.info("Estimating the noise of images.")
            self.noise_var = WhiteNoiseEstimator(self.src).estimate()
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

        self._compress(self.components)

    def _compute_spca(self, coef):
        """
        Algorithm 2 from paper.

        It has been adopted to use ASPIRE-Python's
        cov2d (real) covariance estimation.
        """

        # Compute coefficient vector of mean image at zeroth component
        self.mean_coef_zero = self.mean_coef_est[self.angular_indices == 0]

        # Make the Data matrix (A_k)
        # # Construct A_k, matrix of expansion coefficients a^i_k_q
        # #   for image i, angular index k, radial index q,
        # #   (around eq 31-33)
        # #   Rows radial indices, columns image i.
        # #
        # # We can extract this directly (up to transpose) from
        # #  fb coef matrix where ells == angular_index
        # #  then use the transpose so image stack becomes columns.

        # Initialize a totally empty BlkDiagMatrix, then build incrementally.
        A = BlkDiagMatrix.empty(0, dtype=coef.dtype)

        # Zero angular index is special case of indexing.
        mask = self.basis._indices["ells"] == 0
        A_0 = coef[:, mask] - self.mean_coef_zero
        A.append(A_0)

        # Remaining angular indices have postive and negative entries in real representation.
        for ell in range(
            1, self.basis.ell_max + 1
        ):  # `ell` in this code is `k` from paper
            mask = self.basis._indices["ells"] == ell
            mask_pos = [
                mask[i] and (self.basis._indices["sgns"][i] == +1)
                for i in range(len(mask))
            ]
            mask_neg = [
                mask[i] and (self.basis._indices["sgns"][i] == -1)
                for i in range(len(mask))
            ]

            A.append(coef[:, mask_pos])
            A.append(coef[:, mask_neg])

        if len(A) != len(self.covar_coef_est):
            raise RuntimeError(
                "Data matrix A should have same number of blocks as Covar matrix.",
                f" {len(A)} != {len(self.covar_coef_est)}",
            )

        # For each angular frequency (`ells` in FB code, `k` from paper)
        #   we use the properties of Block Diagonal Matrices to work
        #   on the correspong block.
        eigval_index = 0
        for angular_index, C_k in enumerate(self.covar_coef_est):

            # # Eigen/SVD, covariance block C_k should be symmetric.
            eigvals_k, eigvecs_k = np.linalg.eigh(C_k)

            # Determistically enforce eigen vector sign convention
            eigvecs_k = fix_signs(eigvecs_k)

            # Sort eigvals_k
            sorted_indices = np.argsort(-eigvals_k)
            eigvals_k, eigvecs_k = (
                eigvals_k[sorted_indices],
                eigvecs_k[:, sorted_indices],
            )

            # These are the dense basis indices for this block.
            basis_inds = np.arange(eigval_index, eigval_index + len(eigvals_k))

            # Store the eigvals for this block, note this is a flat array.
            self.eigvals[basis_inds] = eigvals_k

            # Store the eigvecs, note this is a BlkDiagMatrix and is assigned incrementally.
            self.eigvecs[angular_index] = eigvecs_k

            # To compute new expansion coefficients using spca basis
            #   we combine the basis coefs using the eigen decomposition.
            # Note image stack slow moving axis, otherwise this is just a
            #   block by block matrix multiply.
            self.spca_coef[:, basis_inds] = A[angular_index] @ eigvecs_k

            eigval_index += len(eigvals_k)

        # Sanity check we have same dimension of eigvals and basis coefs.
        if eigval_index != self.basis.count:
            raise RuntimeError(
                f"eigvals dimension {eigval_index} != basis coef count {self.basis.count}."
            )

        # Store a map of indices sorted by eigenvalue.
        #  We don't resort then now because this would destroy the block diagonal structure.
        #
        # sorted_indices[i] is the ith most powerful eigendecomposition index
        #
        # We can pass a full or truncated slice of sorted_indices to any array indexed by
        # the coefs.  This is used later for compression and index re-generation.
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

        # Apply linear combination defined by FSPCA (eigvecs)
        c_fspca = x @ self.eigvecs

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

        :param c:  Stack of coefs in the FSPCABasis to be evaluated.
        :return: The (real) coefs representing a stack of images in self.basis
        """

        # apply FSPCA eigenvector to coefs c, yields coefs in self.basis
        eigvecs = self.eigvecs
        if isinstance(eigvecs, BlkDiagMatrix):
            eigvecs = eigvecs.dense()

        # # XXX Legacy code doubles nonzero coefs ?
        # corrected_c = c.copy()
        # corrected_c[:, self.angular_indices!=0] *= 2
        # return corrected_c @ eigvecs.T

        return c @ eigvecs.T

    def _get_compressed_indices(self, n):
        """
        Return the sorted compressed (truncated) indices into the full FSPCA basis.

        Note that we return some number of indices in the real representation (in +- pairs)
        required to cover the `n` components in the complex representation.
        """

        unsigned_components = zip(
            self.angular_indices[self.sorted_indices],
            self.radial_indices[self.sorted_indices],
        )

        # order the components by their importance (occurance based on sorted eigvals)
        #  This isn't exactly right since the eigvals would be sorted by the complex magnitude,
        #    instead of the larger component.
        # Note, that we only use OrderedDict for its key's (ie as an OrderedSet)
        ordered_components = OrderedDict()
        for (k, q) in unsigned_components:
            ordered_components.setdefault((k, q))  # inserts when not exists yet

        # Select the top n (k,q) pairs
        top_components = list(ordered_components)[:n]

        # Now we need to find the locations of both the + and - sgns.
        pos_mask = self.basis._indices["sgns"] == 1
        neg_mask = self.basis._indices["sgns"] == -1
        compressed_indices = []
        for (k, q) in top_components:
            # Compute the locations of coefs we're interested in.
            k_maps = self.angular_indices == k
            q_maps = self.radial_indices == q

            pos_index = np.argmax(k_maps & q_maps & pos_mask)
            compressed_indices.append(pos_index)
            if k > 0:
                neg_index = np.where(k_maps & q_maps & neg_mask)[0][0]
                compressed_indices.append(neg_index)
        return compressed_indices

    # # Noting this is awful, but I'm still trying to work out how we can push the complex arithmetic out and away...
    def _compress(self, n):
        """
        Use the eigendecomposition to select the most powerful
        coefficients.

        Using those coefficients new indice mappings are constructed.

        :param n: Number of components (coef)
        :return: New FSPCABasis instance
        """

        if n >= self.count:
            logger.warning(
                f"Requested compression to {n} components,"
                f" but already {self.count}."
                "  Skipping compression."
            )
            return self

        # Create a deepcopy.
        old = copy.deepcopy(self)

        # Create compressed mapping
        compressed_indices = old._get_compressed_indices(n)
        logger.debug(f"compressed_indices {compressed_indices}")
        self.count = len(compressed_indices)
        logger.debug(f"n {n} compressed count {self.count}")

        # NOTE, no longer blk_diag! ugh
        # Note can copy from old or self, should be same...
        self.eigvals = old.eigvals[compressed_indices]
        if isinstance(old.eigvecs, BlkDiagMatrix):
            old.eigvecs = old.eigvecs.dense()
        self.eigvecs = old.eigvecs[:, compressed_indices]
        self.spca_coef = old.spca_coef[:, compressed_indices]

        self.angular_indices = old.angular_indices[compressed_indices]
        self.radial_indices = old.radial_indices[compressed_indices]
        self.signs_indices = old.signs_indices[compressed_indices]

        self.complex_indices_map = self._get_complex_indices_map()
        self.complex_count = len(self.complex_indices_map)
        self.complex_angular_indices = np.empty(self.complex_count, int)
        self.complex_radial_indices = np.empty(self.complex_count, int)
        for i, key in enumerate(self.complex_indices_map.keys()):
            ang, rad = key
            self.complex_angular_indices[i] = ang
            self.complex_radial_indices[i] = rad

        logger.debug(
            f"complex_radial_indices: {self.complex_radial_indices} {len(self.complex_radial_indices)}"
        )
        logger.debug(
            f"complex_angular_indices: {self.complex_angular_indices} {len(self.complex_angular_indices)}"
        )

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

        ccoef_d = OrderedDict()

        for i in range(self.count):
            ell = self.angular_indices[i]
            q = self.radial_indices[i]
            sgn = self.signs_indices[i]

            ccoef_d.setdefault((ell, q), 0 + 0j)
            if ell == 0:
                ccoef_d[(ell, q)] = coef[:, i]
            elif sgn == 1:
                ccoef_d[(ell, q)] += coef[:, i] / 2.0
            elif sgn == -1:
                ccoef_d[(ell, q)] -= imaginary * coef[:, i] / 2.0
            else:
                raise ValueError("sgns should be +-1")

        for i, k in enumerate(ccoef_d.keys()):
            ccoef[:, i] = ccoef_d[k]

        return ccoef

    def to_real(self, complex_coef):
        """
        Return real valued representation of complex coefficients.
        This can be useful when comparing or implementing methods
        from literature.

        There is a corresponding method, to_complex.

        :param complex_coef: Complex coefficients from this basis.
        :return: Real coefficent representation from this basis.
        """

        if complex_coef.ndim == 1:
            complex_coef = complex_coef.reshape(1, -1)

        if complex_coef.dtype not in (np.complex128, np.complex64):
            raise TypeError("coef provided to to_real should be complex.")

        # Pass through dtype precions, but check and warn if mismatched.
        dtype = real_type(complex_coef.dtype)
        if dtype != self.dtype:
            logger.warning(
                f"Complex coef dtype {complex_coef.dtype} does not match precision of basis.dtype {self.dtype}, returning {dtype}."
            )

        coef = np.zeros((complex_coef.shape[0], self.count), dtype=dtype)

        # map ordered index to (ell, q) key in dict
        keymap = list(self.complex_indices_map.keys())
        for i in range(self.complex_count):
            # retreive index into reals
            pos_i, neg_i = self.complex_indices_map[keymap[i]]
            if self.complex_angular_indices[i] == 0:
                coef[:, pos_i] = complex_coef[:, i].real
            else:
                coef[:, pos_i] = 2.0 * complex_coef[:, i].real
                coef[:, neg_i] = -2.0 * complex_coef[:, i].imag

        return coef

    def rotate(self, coef, radians, refl=None):
        """
        Returns coefs rotated by `radians`.

        :param coef: Basis coefs.
        :param radians: Rotation in radians.
        :param refl: Optional reflect image (bool)
        :return: rotated coefs.
        """

        # Sterrable class rotation expects complex representation of coefficients.
        #  Convert, rotate and convert back to real representation.
        return self.to_real(super().rotate(self.to_complex(coef), radians, refl))

    def calculate_bispectrum(
        self, coef, flatten=False, filter_nonzero_freqs=False, freq_cutoff=None
    ):
        if coef.dtype == real_type(self.dtype):
            coef = self.to_complex(coef)
        return super().calculate_bispectrum(
            coef,
            flatten=flatten,
            filter_nonzero_freqs=filter_nonzero_freqs,
            freq_cutoff=freq_cutoff,
        )

    def eigen_images(self):
        """
        Return the eigen images of the FSPCA basis, evaluated to image space.

        This may be used to implot visualizations of the eigenvectors.

        Ordering corresponds to FSPCA eigvals.
        """

        eigvecs = self.eigvecs
        if isinstance(eigvecs, BlkDiagMatrix):
            eigvecs = eigvecs.dense()

        return self.basis.evaluate(eigvecs.T)
