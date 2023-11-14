import abc
import logging
from collections.abc import Iterable

import numpy as np

from aspire.basis import Basis, Coef, ComplexCoef
from aspire.operators import BlkDiagMatrix
from aspire.utils import LogFilterByCount, complex_type, real_type, trange

logger = logging.getLogger(__name__)


class SteerableBasis2D(Basis, abc.ABC):
    """
    `SteerableBasis2D` is an extension of Basis that is expected to have
    `rotation` (steerable) and `calculate_bispectrum` methods.
    """

    # Default matrix type for basis representation.
    matrix_type = BlkDiagMatrix

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Because they are used for core features of SteerableBasis2D,
        #   cache the indices for positive and negative ells.
        # Note zero is special case.
        self._zero_angular_inds = self.angular_indices == 0
        self._pos_angular_inds = (self.signs_indices == 1) & (self.angular_indices != 0)
        self._neg_angular_inds = self.signs_indices == -1
        self._non_neg_angular_inds = self.signs_indices >= 0
        self._blk_diag_cov_shape = None

        # Centralize indices attributes between FB/PSWF/FLE in SteerableBasis2D
        self.complex_count = self.count - sum(self._neg_angular_inds)
        self.complex_angular_indices = self.angular_indices[self._non_neg_angular_inds]
        self.complex_radial_indices = self.radial_indices[self._non_neg_angular_inds]

        # Attribute for caching the blk_diag shape once known.
        self._blk_diag_cov_shape = None

    def calculate_bispectrum(
        self, complex_coef, flatten=False, filter_nonzero_freqs=False, freq_cutoff=None
    ):
        """
        Calculate bispectrum for a set of coefs in this basis.

        The Bispectum matrix is of shape:
            (count, count, unique_radial_indices)

        where count is the number of complex coefficients.

        :param coef: Coefficients representing a (single) image expanded in this basis.
        :param flatten: Optionally extract symmetric values (tril) and then flatten.
        :param filter_nonzero_freqs: Remove indices corresponding to zero frequency (defaults False).
        :param freq_cutoff: Truncate (zero) high k frequecies above (int) value, defaults off (None).
        :return: Bispectum matrix (complex valued).
        """

        if not isinstance(complex_coef, Coef):
            raise TypeError(f"Expect `Coef` received {type(complex_coef)}.")
        complex_coef = complex_coef.asnumpy()

        # Check shape
        if complex_coef.shape[0] != 1:
            raise ValueError(
                "Due to potentially large sizes, bispectrum is limited to a single set of coefs."
                f"  Passed shape {complex_coef.shape}"
            )

        if complex_coef.shape[1] != self.complex_count:
            raise ValueError(
                "Basis.calculate_bispectrum coefs expected"
                f" to have (complex) count {self.complex_count}, received {complex_coef.shape}."
            )

        # From here just treat complex_coef as 1d vector instead of 1 by count 2d array.
        complex_coef = complex_coef[0]

        if freq_cutoff and freq_cutoff > np.max(self.complex_angular_indices):
            logger.warning(
                f"Bispectrum frequency cutoff {freq_cutoff} outside max {np.max(self.complex_angular_indices)}"
            )

        # Notes, regarding the naming:
        # radial freq indices q in paper/slides, _indices["ks"] in code
        radial_indices = self.complex_radial_indices  # q
        # angular freq indices k in paper/slides, _indices["ells"] in code
        angular_indices = self.complex_angular_indices  # k
        # Compute the set of all unique q in the compressed basis
        #   Note that np.unique is also sorted.
        unique_radial_indices = np.unique(radial_indices)

        # When compressed, we need maps between the basis and uncompressed set of q
        #   to a reduced set of q that remain after compression.
        # One map is provided by self.complex_radial_indices
        #   which maps an index in the basis remaining after compression to a q value.
        # The following code computes a similar but inverted map,
        #   given a q value, find an index into the set of unique q after compression.
        # Also, it is known that the set of q gets sparser with increasing k
        #   but that's ignored here, instead construct a dense
        #   array and filter it later.
        #  The plan is to revisit this code after appropriate coef classes are derived.

        # Default array to fill_value, we can use a value
        # k should never achieve..
        fill_value = self.complex_count**2
        compressed_radial_map = (
            np.ones(np.max(unique_radial_indices) + 1, dtype=int) * fill_value
        )
        for uniq_q_index, q_value in enumerate(unique_radial_indices):
            compressed_radial_map[q_value] = uniq_q_index

        B = np.zeros(
            (self.complex_count, self.complex_count, unique_radial_indices.shape[0]),
            dtype=complex_type(self.dtype),
        )

        logger.info(f"Calculating bispectrum matrix with shape {B.shape}.")

        for ind1 in range(self.complex_count):
            k1 = angular_indices[ind1]
            if freq_cutoff and k1 > freq_cutoff:
                continue
            coef1 = complex_coef[ind1]

            for ind2 in range(self.complex_count):
                k2 = angular_indices[ind2]
                if freq_cutoff and k2 > freq_cutoff:
                    continue
                coef2 = complex_coef[ind2]

                k3 = k1 + k2
                intermodulated_coef_inds = angular_indices == k3

                if np.any(intermodulated_coef_inds):
                    # Get the specific q indices related to feasible k3 angular_indices
                    Q3_ind = radial_indices[intermodulated_coef_inds]

                    if hasattr(self, "compressed") and self.compressed:
                        # Map those Q3_ind values to indices into compressed unique_radial_indices
                        #  by using the compressed_radial_map prepared above.
                        Q3_ind = compressed_radial_map[Q3_ind]

                    Coef3 = complex_coef[intermodulated_coef_inds]

                    B[ind1, ind2, Q3_ind] = coef1 * coef2 * np.conj(Coef3)

        if filter_nonzero_freqs:
            non_zero_freqs = angular_indices != 0
            B = B[non_zero_freqs][:, non_zero_freqs]

        if flatten:
            # B is sym, start by taking lower triangle.
            tril = np.tri(B.shape[0], dtype=bool)
            B = B[tril, :]
            # Then flatten
            B = B.flatten()

        return B

    def rotate(self, coef, radians, refl=None):
        """
        Returns coefs rotated counter-clockwise by `radians`.

        :param coef: Basis coefs.
        :param radians: Rotation in radians.
        :param refl: Optional reflect image (about y=0) (bool)
        :return: rotated coefs.
        """

        if not isinstance(coef, Coef):
            raise TypeError(f"`coef` must be `Coef` instance, received {type(coef)}.")

        coef = coef.asnumpy()

        # Covert radians to a broadcastable shape
        if isinstance(radians, Iterable):
            radians = np.array(radians, dtype=self.dtype)
            if radians.ndim < 2:
                radians = radians.reshape(-1, 1)
            else:
                radians = np.expand_dims(radians, axis=-1)

            if radians.size != np.prod(coef.shape[:-1]):
                raise RuntimeError(
                    f"`rotate` call `radians` {radians.shape} does not match"
                    f" `coef` {coef.shape[:-1]}."
                )
        # else: radians can be a constant

        assert (
            self.count == coef.shape[-1]
        ), "Number of coefficients must match self.count."

        # self.angular_indices are `ks`
        # For all coef in stack,
        #   compute the ks * radian used in the trig functions
        ks_rad = np.atleast_2d(self.angular_indices * radians)
        ks_pos = ks_rad[..., self._pos_angular_inds]
        ks_neg = ks_rad[..., self._neg_angular_inds]

        # Slice the coef on postive and negative ells
        coef_zer = coef[..., self._zero_angular_inds]
        coef_pos = coef[..., self._pos_angular_inds]
        coef_neg = coef[..., self._neg_angular_inds]

        # Handle zero case and avoid mutating the original array
        coef = np.empty_like(coef)
        coef[..., self._zero_angular_inds] = coef_zer

        # refl
        if refl is not None:
            if isinstance(refl, np.ndarray):
                assert len(refl) == len(coef)
            # else: refl can be a constant
            # negate the coefs corresponding to negative ells
            coef_neg[refl] = coef_neg[refl] * -1

        # Apply formula
        coef[..., self._pos_angular_inds] = coef_pos * np.cos(
            ks_pos
        ) + coef_neg * np.sin(ks_neg)
        coef[..., self._neg_angular_inds] = coef_neg * np.cos(
            ks_neg
        ) - coef_pos * np.sin(ks_pos)

        return Coef(self, coef)

    def complex_rotate(self, complex_coef, radians, refl=None):
        """
        Returns complex coefs rotated counter-clockwise by `radians`.

        This implementation uses the complex exponential.
        It is kept in the code for documentation and
        reference purposes.

        To invoke in code:

        self.to_real(
            self.complex_rotate(
                self.to_complex(coef), radians, refl)
            )
        )

        :param complex_coef: Basis coefs (in complex representation).
        :param radians: Rotation in radians.
        :param refl: Optional reflect image (about y=0) (bool)
        :return: rotated (complex) coefs.
        """

        # Covert radians to a broadcastable shape
        if isinstance(radians, np.ndarray):
            if len(radians) != len(complex_coef):
                raise RuntimeError(
                    "`rotate` call `radians` length cannot broadcast with"
                    f" `complex_coef` {len(complex_coef)} != {len(radians)}"
                )
            radians = radians.reshape(-1, 1)
        # else: radians can be a constant

        ks = self.complex_angular_indices
        assert len(ks) == complex_coef.shape[-1]

        # Don't mutate the input coef array (danger)
        _complex_coef = complex_coef.copy()

        # refl
        if refl is not None:
            if isinstance(refl, np.ndarray):
                assert len(refl) == len(complex_coef)
            # else: refl can be a constant
            # get the coefs corresponding to -ks , aka "ells"
            _complex_coef[refl] = np.conj(complex_coef[refl])

        _complex_coef = _complex_coef * np.exp(1j * ks * radians)

        return _complex_coef

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

        return self.evaluate_t(self.evaluate(coef).shift(shifts))

    @property
    def blk_diag_cov_shape(self):
        """
        Return the `BlkDiagMatrix` partition shapes.

        If the shape has already been cached,
        returns cached value.  Otherwise, will
        compute the shape and cache in this instance.
        """
        # Compute the _blk_diag_cov_shape as needed.
        if self._blk_diag_cov_shape is None:
            blks = []
            for ell in range(self.ell_max + 1):
                sgns = (1,) if ell == 0 else (1, -1)
                for _ in sgns:
                    blks.append(
                        [
                            self.k_max[ell],
                        ]
                        * 2
                    )
            self._blk_diag_cov_shape = np.array(blks)

        # Return the cached shape
        return self._blk_diag_cov_shape

    # This is included for completion, but is not being adopted yet.
    def indices_mask(self, **kwargs):
        """
        Given `radial=` or `angular=` expressions, return (`count`,)
        shaped mask where values satisfying the expression are `True`.

        Examples:
            No args yield all indices.
            `angular=0 creates a mask for selecting coefficients with zero angular indices.
            `angular=1, radial=2` selects coefficients satisfying angular index of 1 _and_ radial index of 2.
            More advanced operations can combine indices attributes.
             `angular=self.angular_indices>=0, radial=r` selects coefficients with non negative angular indices and some radial index `r`.

        :return: Boolen mask of shape (`count`,).
            Intended to be broadcast with `Coef` containers.
        """

        radial = kwargs.get("radial", None)
        angular = kwargs.get("angular", None)
        signs = kwargs.get("signs", None)

        # slowly construct the map
        signs_mask = np.zeros(self.count, dtype=bool)
        radial_mask = signs_mask.copy()
        angular_mask = signs_mask.copy()

        if radial is None:
            radial_mask[:] = True
        else:
            for k in np.atleast_1d(radial):
                radial_mask[self.radial_indices == k] = True

        if angular is None:
            angular_mask[:] = True
        else:
            for el in np.atleast_1d(angular):
                angular_mask[self.angular_indices == el] = True

        if signs is None:
            signs_mask[:] = True
        else:
            for s in np.atleast_1d(signs):
                signs_mask[self.signs_indices == s] = True

        mask = radial_mask & angular_mask & signs_mask

        return mask

    def to_real(self, complex_coef):
        """
        Return real valued representation of complex coefficients.
        This can be useful when comparing, prototyping, or
        implementing methods from literature.

        There is a corresponding method, `to_complex`.

        :param complex_coef: Complex `Coef` from this basis.
        :return: Real `Ceof` representation from this basis.
        """

        if not isinstance(complex_coef, ComplexCoef):
            raise TypeError(
                f"complex_coef should be instance of `Coef`, received {type(complex_coef)}."
            )

        if complex_coef.dtype not in (np.complex128, np.complex64):
            raise TypeError("coef provided to to_real should be complex.")

        # Pass through dtype precisions, but check and warn if mismatched.
        dtype = real_type(complex_coef.dtype)
        if dtype != self.dtype:
            logger.warning(
                f"Complex coef dtype {complex_coef.dtype} does not match precision of basis.dtype {self.dtype}, returning {dtype}."
            )

        coef = np.zeros((*complex_coef.stack_shape, self.count), dtype=dtype)
        complex_coef = complex_coef.asnumpy()

        ind = 0
        idx = np.arange(self.k_max[0], dtype=int)
        ind += np.size(idx)
        ind_pos = ind

        coef[..., idx] = complex_coef[..., idx].real

        for ell in range(1, self.ell_max + 1):
            idx = ind + np.arange(self.k_max[ell], dtype=int)
            idx_pos = ind_pos + np.arange(self.k_max[ell], dtype=int)
            idx_neg = idx_pos + self.k_max[ell]

            c = complex_coef[..., idx]
            coef[..., idx_pos] = 2.0 * np.real(c)
            coef[..., idx_neg] = -2.0 * np.imag(c)

            ind += np.size(idx)
            ind_pos += 2 * self.k_max[ell]

        return Coef(self, coef)

    def to_complex(self, coef):
        """
        Return complex valued representation of complex coefficients.
        This can be useful when comparing, prototyping, or
        implementing methods from literature.

        There is a corresponding method, `to_real`.

        :param coef: Real `Coef` from this basis.
        :return: `ComplexCoef` representation from this basis.
        """

        if not isinstance(coef, Coef):
            raise TypeError(
                f"coef should be instance of `Coef`, received {type(coef)}."
            )

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

        complex_coef = np.zeros((*coef.stack_shape, self.complex_count), dtype=dtype)
        coef = coef.asnumpy()

        ind = 0
        idx = np.arange(self.k_max[0], dtype=int)
        ind += np.size(idx)

        complex_coef[..., idx] = coef[..., idx]

        for ell in range(1, self.ell_max + 1):
            idx = ind + np.arange(self.k_max[ell], dtype=int)
            complex_coef[..., idx] = (
                coef[..., self._pos[idx]] - imaginary * coef[..., self._neg[idx]]
            ) / 2.0

            ind += np.size(idx)

        return ComplexCoef(self, complex_coef)

    # `abstractmethod` enforces when a new subclass of
    # `SteerableBasis2D` is created that this method is explicitly
    # implemented.  This is intended to encourage future basis authors
    # to consider this method for their application.
    @abc.abstractmethod
    def filter_to_basis_mat(self, f, method="evaluate_t", truncate=True):
        """
        Convert a filter into a basis operator representation.

        :param f: `Filter` object, usually a `CTFFilter`.
        :param method: `evaluate_t` or `expand`.
        :param truncate: Optionally, truncate dense matrix to BlkDiagMatrix.
            Defaults to True.

        :return: Representation of filter as `basis` operator.
            Return type will be based on the class's `matrix_type`.
        """
        # evaluate_t is not as accurate, but much much faster...
        if method == "evaluate_t":
            expand_method = self.evaluate_t
        elif method == "expand":
            expand_method = self.expand
        else:
            raise NotImplementedError(
                "`filter_to_basis_mat` method {method} not supported."
                "  Try `evaluate_t` or `expand`."
            )

        coef = Coef(self, np.eye(self.count, dtype=self.dtype))
        img = coef.evaluate()

        # Expansion can fail for some filters on specific basis vectors.
        # Loop over the expanding the filtered basis vectors one by one,
        # zero-ing failed vectors.
        filt = np.zeros((self.count, self.count), self.dtype)
        with LogFilterByCount(logger, 1):
            for i in trange(self.count):
                try:
                    filt[i] = expand_method(img[i].filter(f)).asnumpy()[0]
                except Exception:
                    logger.warning(
                        f"Failed to expand basis vector {i} after filter {f}."
                    )

        # Optionally truncate off block elements to zero.
        if truncate:
            filt = BlkDiagMatrix.from_dense(
                filt,
                self.blk_diag_cov_shape,
            )

        return filt
