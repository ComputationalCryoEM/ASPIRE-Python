import logging

import numpy as np
from scipy.special import jv

from aspire.basis import FBBasisMixin, SteerableBasis2D
from aspire.basis.basis_utils import unique_coords_nd
from aspire.utils import roll_dim, unroll_dim
from aspire.utils.matlab_compat import m_flatten, m_reshape

logger = logging.getLogger(__name__)


class FBBasis2D(SteerableBasis2D, FBBasisMixin):
    """
    Define a derived class using the Fourier-Bessel basis for mapping 2D images

    The expansion coefficients of 2D images on this basis are obtained by
    the least squares method. The algorithm is described in the publication:
    Z. Zhao, A. Singer, Fourier-Bessel Rotational Invariant Eigenimages,
    The Journal of the Optical Society of America A, 30 (5), pp. 871-877 (2013).

    """

    # TODO: Methods that return dictionaries should return useful objects instead
    def __init__(self, size, ell_max=None, dtype=np.float32):
        """
        Initialize an object for the 2D Fourier-Bessel basis class

        :param size: The size of the vectors for which to define the basis.
            May be a 2-tuple or an integer, in which case a square basis is assumed.
            Currently only square images are supported.
        :ell_max: The maximum order ell of the basis elements. If no input
            (= None), it will be set to np.Inf and the basis includes all
            ell such that the resulting basis vectors are concentrated
            below the Nyquist frequency (default Inf).
        """

        if isinstance(size, int):
            size = (size, size)
        ndim = len(size)
        assert ndim == 2, "Only two-dimensional basis functions are supported."
        assert len(set(size)) == 1, "Only square domains are supported."
        super().__init__(size, ell_max, dtype=dtype)

    def _build(self):
        """
        Build the internal data structure to 2D Fourier-Bessel basis
        """
        logger.info(
            "Expanding 2D images in a spatial-domain Fourier–Bessel"
            " basis using the direct method."
        )

        # get upper bound of zeros, ells, and ks  of Bessel functions
        self._calc_k_max()

        # calculate total number of basis functions
        self.count = self.k_max[0] + sum(2 * self.k_max[1:])

        # obtain a 2D grid to represent basis functions
        self.basis_coords = unique_coords_nd(self.nres, self.ndim, dtype=self.dtype)

        # generate 1D indices for basis functions
        self._compute_indices()

        # get normalized factors
        self.radial_norms, self.angular_norms = self.norms()

        # precompute the basis functions in 2D grids
        self._precomp = self._precomp()

    def _compute_indices(self):
        """
        Create the indices for each basis function
        """
        indices_ells = np.zeros(self.count, dtype=int)
        indices_ks = np.zeros(self.count, dtype=int)
        indices_sgns = np.zeros(self.count, dtype=int)

        # We'll also generate a mapping for complex construction
        self.complex_count = sum(self.k_max)
        # These map indices in complex array to pair of indices in real array
        self._pos = np.zeros(self.complex_count, dtype=int)
        self._neg = np.zeros(self.complex_count, dtype=int)

        i = 0
        ci = 0
        for ell in range(self.ell_max + 1):
            sgns = (1,) if ell == 0 else (1, -1)
            ks = np.arange(0, self.k_max[ell])

            for sgn in sgns:
                rng = np.arange(i, i + len(ks))
                indices_ells[rng] = ell
                indices_ks[rng] = ks
                indices_sgns[rng] = sgn

                if sgn == 1:
                    self._pos[ci + ks] = rng
                elif sgn == -1:
                    self._neg[ci + ks] = rng

                i += len(ks)

            ci += len(ks)

        self.angular_indices = indices_ells
        self.radial_indices = indices_ks
        self.signs_indices = indices_sgns

    def _precomp(self):
        """
        Precompute the basis functions at defined sample points
        """

        r_unique = self.basis_coords["r_unique"]
        ang_unique = self.basis_coords["ang_unique"]

        ind_radial = 0
        ind_ang = 0

        radial = np.zeros(shape=(len(r_unique), np.sum(self.k_max)), dtype=self.dtype)
        ang = np.zeros(
            shape=(ang_unique.shape[-1], 2 * self.ell_max + 1), dtype=self.dtype
        )

        for ell in range(0, self.ell_max + 1):
            for k in range(1, self.k_max[ell] + 1):
                # Only normalized by the radial part of basis function
                radial[:, ind_radial] = (
                    jv(ell, self.r0[ell][k - 1] * r_unique)
                    / self.radial_norms[ind_radial]
                )
                ind_radial += 1

            sgns = (1,) if ell == 0 else (1, -1)
            for sgn in sgns:
                fn = np.cos if sgn == 1 else np.sin
                ang[:, ind_ang] = fn(ell * ang_unique)
                ind_ang += 1

        return {"radial": radial, "ang": ang}

    def norms(self):
        """
        Calculate the normalized factors of basis functions
        """
        radial_norms = np.zeros(np.sum(self.k_max), dtype=self.dtype)
        angular_norms = np.zeros(np.sum(self.k_max), dtype=self.dtype)
        norm_fn = self.basis_norm_2d

        i = 0
        for ell in range(0, self.ell_max + 1):
            for k in range(1, self.k_max[ell] + 1):
                radial_norms[i], angular_norms[i] = norm_fn(ell, k)
                i += 1

        return radial_norms, angular_norms

    def basis_norm_2d(self, ell, k):
        """
        Calculate the normalized factors from radial and angular parts of a specified basis function
        """
        rad_norm = (
            np.abs(jv(ell + 1, self.r0[ell][k - 1]))
            * np.sqrt(1 / 2.0)
            * self.nres
            / 2.0
        )
        ang_norm = np.sqrt(np.pi)
        if ell == 0:
            ang_norm *= np.sqrt(2)

        return rad_norm, ang_norm

    def _evaluate(self, v):
        """
        Evaluate coefficients in standard 2D coordinate basis from those in FB basis

        :param v: A coefficient vector (or an array of coefficient vectors) to
            be evaluated. The last dimension must equal `self.count`.
        :return: The evaluation of the coefficient vector(s) `v` for this basis.
            This is an array whose last dimensions equal `self.sz` and the remaining
            dimensions correspond to first dimensions of `v`.
        """
        # Transpose here once, instead of several times below  #RCOPT
        v = v.reshape(-1, self.count).T

        r_idx = self.basis_coords["r_idx"]
        ang_idx = self.basis_coords["ang_idx"]
        mask = m_flatten(self.basis_coords["mask"])

        ind = 0
        ind_radial = 0
        ind_ang = 0

        x = np.zeros(shape=tuple([np.prod(self.sz)] + list(v.shape[1:])), dtype=v.dtype)
        for ell in range(0, self.ell_max + 1):
            k_max = self.k_max[ell]
            idx_radial = ind_radial + np.arange(0, k_max, dtype=int)

            # include the normalization factor of angular part
            ang_nrms = self.angular_norms[idx_radial]
            radial = self._precomp["radial"][:, idx_radial]
            radial = radial / ang_nrms

            sgns = (1,) if ell == 0 else (1, -1)
            for _ in sgns:
                ang = self._precomp["ang"][:, ind_ang]
                ang_radial = np.expand_dims(ang[ang_idx], axis=1) * radial[r_idx]
                idx = ind + np.arange(0, k_max, dtype=int)
                x[mask] += ang_radial @ v[idx]
                ind += len(idx)
                ind_ang += 1

            ind_radial += len(idx_radial)

        x = x.T.reshape(-1, *self.sz)  # RCOPT

        return x

    def _evaluate_t(self, v):
        """
        Evaluate coefficient in FB basis from those in standard 2D coordinate basis

        :param v: The coefficient array to be evaluated. The last dimensions
            must equal `self.sz`.
        :return: The evaluation of the coefficient array `v` in the dual basis
            of `basis`. This is an array of vectors whose last dimension equals
            `self.count` and whose first dimensions correspond to
            first dimensions of `v`.
        """
        v = v.T
        x, sz_roll = unroll_dim(v, self.ndim + 1)
        x = m_reshape(
            x, new_shape=tuple([np.prod(self.sz)] + list(x.shape[self.ndim :]))
        )

        r_idx = self.basis_coords["r_idx"]
        ang_idx = self.basis_coords["ang_idx"]
        mask = m_flatten(self.basis_coords["mask"])

        ind = 0
        ind_radial = 0
        ind_ang = 0

        v = np.zeros(shape=tuple([self.count] + list(x.shape[1:])), dtype=v.dtype)
        for ell in range(0, self.ell_max + 1):
            k_max = self.k_max[ell]
            idx_radial = ind_radial + np.arange(0, k_max)
            # include the normalization factor of angular part
            ang_nrms = self.angular_norms[idx_radial]
            radial = self._precomp["radial"][:, idx_radial]
            radial = radial / ang_nrms

            sgns = (1,) if ell == 0 else (1, -1)
            for _ in sgns:
                ang = self._precomp["ang"][:, ind_ang]
                ang_radial = np.expand_dims(ang[ang_idx], axis=1) * radial[r_idx]
                idx = ind + np.arange(0, k_max)
                v[idx] = ang_radial.T @ x[mask]
                ind += len(idx)
                ind_ang += 1

            ind_radial += len(idx_radial)

        v = roll_dim(v, sz_roll)
        return v.T  # RCOPT

    def calculate_bispectrum(
        self, coef, flatten=False, filter_nonzero_freqs=False, freq_cutoff=None
    ):
        """
        Calculate bispectrum for a set of coefs in this basis.

        The Bispectum matrix is of shape:
            (count, count, unique_radial_indices)

        where count is the number of complex coefficients.

        :param coef: Coefficients representing a (single) image expanded in this basis.
        :param flatten: Optionally extract symmetric values (tril) and then flatten.
        :param freq_cutoff: Truncate (zero) high k frequecies above (int) value, defaults off (None).
        :return: Bispectum matrix (complex valued).
        """

        # Bispectrum implementation expects the complex representation of coefficients.
        complex_coef = self.to_complex(coef)

        return super().calculate_bispectrum(
            complex_coef,
            flatten=flatten,
            filter_nonzero_freqs=filter_nonzero_freqs,
            freq_cutoff=freq_cutoff,
        )

    def filter_to_basis_mat(self, *args, **kwargs):
        """
        See `SteerableBasis2D.filter_to_basis_mat`.
        """
        return super().filter_to_basis_mat(*args, **kwargs)
