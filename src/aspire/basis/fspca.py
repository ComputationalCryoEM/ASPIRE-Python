import logging
from collections import OrderedDict

import numpy as np

from aspire.basis import Coef, ComplexCoef, FFBBasis2D, SteerableBasis2D
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

    def __init__(
        self, src, basis=None, noise_var=None, components=None, batch_size=512
    ):
        """

        :param src: Source instance
        :param basis: Optional Fourier Bessel Basis (usually FFBBasis2D)
        :param components: Optionally assign number of principal components
            to use for the FSPCA basis.
            Default value of `None` will use `self.basis.count`.
        :param noise_var: Optionally assign noise variance.
            Default value of `None` will estimate noise with WhiteNoiseEstimator.
            Use 0 when using clean images so cov2d skips applying noisy covar coefs..
        :param batch_size: Batch size for computing basis coefficients.
            `batch_size` is also passed to BatchedRotCov2D.
        """

        self.src = src
        self.batch_size = batch_size

        # Automatically generate basis if needed.
        if basis is None:
            basis = FFBBasis2D((self.src.L,) * 2, dtype=self.src.dtype)
        self.basis = basis

        # Components are used for `compress` during `build`.
        self.components = components or self.basis.count
        self._check_components()

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
        self.signs_indices = self.basis.signs_indices
        self.complex_angular_indices = self.basis.complex_angular_indices
        self.complex_radial_indices = self.basis.complex_radial_indices

        self.complex_indices_map = self._get_complex_indices_map()
        assert (
            len(self.complex_indices_map) == self.complex_count
        ), f"{len(self.complex_indices_map)} != {self.complex_count}"

        # Map Cached SteerableBasis2D index maps
        self._zero_angular_inds = self.basis._zero_angular_inds
        self._pos_angular_inds = self.basis._pos_angular_inds
        self._neg_angular_inds = self.basis._neg_angular_inds

        self.noise_var = noise_var  # noise_var is handled during `build` call.

        self.build()

    def _check_components(self):
        """
        Check that our (compressed) count is not larger than our basis count.
        """
        if self.components > self.basis.count:
            raise ValueError(
                f"Provided components {self.components} > {self.basis.count} basis coefficients."
                "  Reduce components."
            )

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

        if self.noise_var is None:
            from aspire.noise import WhiteNoiseEstimator

            logger.info("Estimating the noise of images.")
            self.noise_var = WhiteNoiseEstimator(self.src).estimate()
        logger.info(f"Setting noise_var={self.noise_var}")

        # Import BatchedRotCov2D here to prevent circular imports.
        from aspire.covariance import BatchedRotCov2D

        cov2d = BatchedRotCov2D(
            src=self.src, basis=self.basis, batch_size=self.batch_size
        )
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
        self.mean_coef_est = cov2d.get_mean()
        self.covar_coef_est = cov2d.get_covar(
            mean_coef=self.mean_coef_est,
            noise_var=self.noise_var,
            covar_est_opt=covar_opt,
        )

        # Create the arrays to be packed by _compute_spca
        self._eigvals = np.zeros(self.basis.count, dtype=self.dtype)

        self.eigvecs = BlkDiagMatrix.empty(2 * self.basis.ell_max + 1, dtype=self.dtype)

        # Perform the PCA over batches, storing the compressed coefficients.
        self._compute_spca()

        #  Complete compression by mutating class
        self._compress()

    def _compute_spca(self):
        """
        Algorithm 2 from paper.

        It has been adopted to use ASPIRE-Python's
        cov2d (real) covariance estimation.
        """

        # -- Compute the spectrum blockwise. --
        # For each angular frequency (`ells` in FB code, `k` from paper)
        #   we use the properties of Block Diagonal Matrices to work
        #   on the correspong block.
        eigval_index = 0
        basis_inds = []
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
            _basis_inds = np.arange(eigval_index, eigval_index + len(eigvals_k))
            basis_inds.append(_basis_inds)

            # Store the eigvals for this block, note this is a flat array.
            self._eigvals[_basis_inds] = eigvals_k

            # Store the eigvecs, note this is a BlkDiagMatrix and is assigned incrementally.
            self.eigvecs[angular_index] = eigvecs_k

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
        self.sorted_indices = np.argsort(-np.abs(self._eigvals))

        compressed_indices = self._get_compressed_indices()

        self.spca_coef = np.zeros(
            (self.src.n, len(compressed_indices)), dtype=self.dtype
        )

        # Compute coefficient vector of mean image at zeroth component
        self.mean_coef_zero = self.mean_coef_est.asnumpy()[0][self.angular_indices == 0]

        # Define mask for zero angular mode, used in loop below
        zero_ell_mask = self.basis.angular_indices == 0

        # Apply Data matrix batchwise
        num_batches = (self.src.n + self.batch_size - 1) // self.batch_size
        for i in range(num_batches):
            # Compute the coefficients for this batch
            start = i * self.batch_size
            finish = min((i + 1) * self.batch_size, self.src.n)
            batch_coef = self.basis.evaluate_t(self.src.images[start:finish])
            batch_coef = batch_coef.asnumpy()

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
            A = BlkDiagMatrix.empty(0, dtype=batch_coef.dtype)

            # Zero angular index is special case of indexing.
            A_0 = batch_coef[:, zero_ell_mask] - self.mean_coef_zero
            A.append(A_0)

            # Remaining angular indices have postive and negative entries in real representation.
            for ell in range(
                1, self.basis.ell_max + 1
            ):  # `ell` in this code is `k` from paper
                mask_ell = self.basis.angular_indices == ell
                mask_pos = mask_ell & (self.basis.signs_indices == +1)
                mask_neg = mask_ell & (self.basis.signs_indices == -1)

                A.append(batch_coef[:, mask_pos])
                A.append(batch_coef[:, mask_neg])

            if len(A) != len(self.covar_coef_est):
                raise RuntimeError(
                    "Data matrix A should have same number of blocks as Covar matrix.",
                    f" {len(A)} != {len(self.covar_coef_est)}",
                )

            # -- Compute new FSPCA coefficients. --
            # For each batch
            #   For each angular frequency (`ells` in FB code, `k` from paper)
            #     Use the properties of Block Diagonal Matrices to work
            #     on the correspong block.
            blk_spca_coef = np.empty_like(batch_coef)
            for angular_index, a_blk in enumerate(A):
                # To compute new expansion coefficients using spca basis
                #   we combine the basis coefs using the eigen decomposition.
                # Note image stack slow moving axis, otherwise this is just a
                #   block by block matrix multiply.
                blk_spca_coef[:, basis_inds[angular_index]] = (
                    a_blk @ self.eigvecs[angular_index]
                )

            # Assign truncated block to global spca_coef
            self.spca_coef[start:finish, :] = blk_spca_coef[:, compressed_indices]

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
        if not isinstance(x, Coef):
            raise TypeError(f"'x' should be `Coef` instance, received {type(x)}.")

        # Apply linear combination defined by FSPCA (eigvecs)
        c_fspca = x.asnumpy() @ self.eigvecs

        assert c_fspca.shape == (x.shape[0], self.count)

        return Coef(self, c_fspca)

    def evaluate_to_image_basis(self, c):
        """
        Take FSPCA coefs and evaluate as image in the standard coordinate basis.

        :param c:  Stack of coefs in the FSPCABasis to be evaluated.
        :return: The Image instance representing a stack of images in the
            standard 2D coordinate basis..
        """
        if not isinstance(c, Coef):
            raise TypeError(f"'c' should be `Coef` instance, received {type(c)}.")

        c_fb = self.evaluate(c)

        return self.basis.evaluate(c_fb)

    def evaluate(self, c):
        """
        Take FSPCA coefs and evaluate to Fourier Bessel (self.basis) ceofs.

        :param c:  Stack of coefs in the FSPCABasis to be evaluated.
        :return: The (real) coefs representing a stack of images in self.basis
        """

        if not isinstance(c, Coef):
            raise TypeError(f"'c' should be `Coef` instance, received {type(c)}.")

        # apply FSPCA eigenvector to coefs c, yields coefs in self.basis
        eigvecs = self.eigvecs
        if isinstance(eigvecs, BlkDiagMatrix):
            eigvecs = eigvecs.dense()

        # # XXX Legacy code doubles nonzero coefs ?
        # corrected_c = c.copy()
        # corrected_c[:, self.angular_indices!=0] *= 2
        # return corrected_c @ eigvecs.T

        return Coef(self.basis, c @ eigvecs.T)

    # TODO: Python>=3.8 @cached_property
    def _get_compressed_indices(self):
        """
        Return the sorted compressed (truncated) indices into the full FSPCA basis.

        Note that we return some number of indices in the real representation (in +- pairs)
        required to cover the `self.components` in the complex representation.
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
        for k, q in unsigned_components:
            ordered_components.setdefault((k, q))  # inserts when not exists yet

        # Select the top n (k,q) pairs
        top_components = list(ordered_components)[: self.components]

        # Now we need to find the locations of both the + and - sgns.
        pos_mask = self.basis.signs_indices == 1
        neg_mask = self.basis.signs_indices == -1
        compressed_indices = []
        for k, q in top_components:
            # Compute the locations of coefs we're interested in.
            k_maps = self.angular_indices == k
            q_maps = self.radial_indices == q

            pos_index = np.argmax(k_maps & q_maps & pos_mask)
            compressed_indices.append(pos_index)
            if k > 0:
                neg_index = np.where(k_maps & q_maps & neg_mask)[0][0]
                compressed_indices.append(neg_index)
        return compressed_indices

    def _compress(self):
        """
        Use the eigendecomposition to select the most powerful
        coefficients.

        Using those coefficients new indice mappings are constructed.

        Mutates `self`.

        :param n: Number of components (coef)
        """

        if self.components >= self.count:
            logger.warning(
                f"Requested compression to {self.components} components,"
                f" but already {self.count}."
                "  Skipping compression."
            )
            return self

        # Create compressed mapping
        compressed_indices = self._get_compressed_indices()
        self.count = len(compressed_indices)

        self._eigvals = self._eigvals[compressed_indices]
        if isinstance(self.eigvecs, BlkDiagMatrix):
            self.eigvecs = self.eigvecs.dense()
        self.eigvecs = self.eigvecs[:, compressed_indices]

        self.angular_indices = self.angular_indices[compressed_indices]
        self.radial_indices = self.radial_indices[compressed_indices]
        self.signs_indices = self.signs_indices[compressed_indices]

        self.complex_indices_map = self._get_complex_indices_map()
        self.complex_count = len(self.complex_indices_map)
        self.complex_angular_indices = np.empty(self.complex_count, int)
        self.complex_radial_indices = np.empty(self.complex_count, int)
        for i, key in enumerate(self.complex_indices_map.keys()):
            ang, rad = key
            self.complex_angular_indices[i] = ang
            self.complex_radial_indices[i] = rad

    def to_complex(self, coef):
        """
        Return complex valued representation of coefficients.
        This can be useful when comparing or implementing methods
        from literature.

        There is a corresponding method, to_real.

        :param coef: Coefficients from this basis.
        :return: Complex coeficent representation from this basis.
        """
        if not isinstance(coef, Coef):
            raise TypeError(f"'coef' should be `Coef` instance, received {type(coef)}.")
        coef = coef.asnumpy()

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

        return ComplexCoef(self, ccoef)

    def to_real(self, complex_coef):
        """
        Return real valued representation of complex coefficients.
        This can be useful when comparing or implementing methods
        from literature.

        There is a corresponding method, to_complex.

        :param complex_coef: Complex coefficients from this basis.
        :return: Real coefficient representation from this basis.
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

        return Coef(self, coef)

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

    @property
    def eigvals(self):
        """
        Return the eigenvals of FSPCABasis as Numpy array.
        """
        return self._eigvals

    def eigen_images(self):
        """
        Return the eigen images of the FSPCA basis, evaluated to image space.

        This may be used to implot visualizations of the eigenvectors.

        Ordering corresponds to FSPCA eigvals.
        """

        eigvecs = self.eigvecs
        if isinstance(eigvecs, BlkDiagMatrix):
            eigvecs = eigvecs.dense()

        return Coef(self.basis, eigvecs.T).evaluate()

    def shift(self, coef, shifts):
        """
        Returns coefs shifted by `shifts`.

        This will transform to real cartesian space, shift,
        and transform back to Polar Fourier-Bessel space.

        :param coef: Basis coefs.
        :param shifts: Shifts in pixels (x,y). Shape (1,2) or (len(coef), 2).
        :return: coefs of shifted images.
        """

        shifts = np.atleast_2d(np.array(shifts))
        if shifts.ndim != 2:
            raise ValueError("`shifts` should be a one or two dimensional array.")
        if shifts.shape[1] != 2 or shifts.shape[0] not in (1, len(coef)):
            raise ValueError(
                "`shifts` should be shape (1,2) or (len(coef),2),"
                f" received {shifts.shape}."
            )

        # This transforms FSPCA->FB->Image->FB->FSPCA
        return self.expand_from_image_basis(
            self.evaluate_to_image_basis(coef).shift(shifts)
        )

    def filter_to_basis_mat(self, f):
        """
        Convert a filter into a basis representation.

        :param f: `Filter` object, usually a `CTFFilter`.

        :return: Representation of filter in `basis`.
            Return type will be based on the class's `matrix_type`.
        """
        # This is possible to implement, but there are no current use cases.
        raise NotImplementedError("Not currently implemented for compressed basis.")
