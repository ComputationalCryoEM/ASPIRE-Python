import logging

import numpy as np
from scipy.special import jv

from aspire.basis import Basis
from aspire.basis.basis_utils import unique_coords_nd
from aspire.utils import ensure, roll_dim, unroll_dim
from aspire.utils.matlab_compat import m_flatten, m_reshape

logger = logging.getLogger(__name__)


class FBBasis2D(Basis):
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
            Currently only square images are supported.
        :ell_max: The maximum order ell of the basis elements. If no input
            (= None), it will be set to np.Inf and the basis includes all
            ell such that the resulting basis vectors are concentrated
            below the Nyquist frequency (default Inf).
        """

        ndim = len(size)
        ensure(ndim == 2, "Only two-dimensional basis functions are supported.")
        ensure(len(set(size)) == 1, "Only square domains are supported.")
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
        self._getfbzeros()

        # calculate total number of basis functions
        self.count = self.k_max[0] + sum(2 * self.k_max[1:])

        # obtain a 2D grid to represent basis functions
        self.basis_coords = unique_coords_nd(self.nres, self.ndim, dtype=self.dtype)

        # generate 1D indices for basis functions
        self._indices = self.indices()

        # get normalized factors
        self.radial_norms, self.angular_norms = self.norms()

        # precompute the basis functions in 2D grids
        self._precomp = self._precomp()

    def indices(self):
        """
        Create the indices for each basis function
        """
        indices_ells = np.zeros(self.count, dtype=np.int)
        indices_ks = np.zeros(self.count, dtype=np.int)
        indices_sgns = np.zeros(self.count, dtype=np.int)

        i = 0
        for ell in range(self.ell_max + 1):
            sgns = (1,) if ell == 0 else (1, -1)
            ks = range(0, self.k_max[ell])

            for sgn in sgns:
                rng = range(i, i + len(ks))
                indices_ells[rng] = ell
                indices_ks[rng] = ks
                indices_sgns[rng] = sgn

                i += len(rng)

        return {"ells": indices_ells, "ks": indices_ks, "sgns": indices_sgns}

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
                    jv(ell, self.r0[k - 1, ell] * r_unique)
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
            np.abs(jv(ell + 1, self.r0[k - 1, ell]))
            * np.sqrt(1 / 2.0)
            * self.nres
            / 2.0
        )
        ang_norm = np.sqrt(np.pi)
        if ell == 0:
            ang_norm *= np.sqrt(2)

        return rad_norm, ang_norm

    def evaluate(self, v):
        """
        Evaluate coefficients in standard 2D coordinate basis from those in FB basis

        :param v: A coefficient vector (or an array of coefficient vectors) to
            be evaluated. The last dimension must equal `self.count`.
        :return: The evaluation of the coefficient vector(s) `v` for this basis.
            This is an array whose last dimensions equal `self.sz` and the remaining
            dimensions correspond to first dimensions of `v`.
        """

        if v.dtype != self.dtype:
            logger.warning(
                f"{self.__class__.__name__}::evaluate"
                f" Inconsistent dtypes v: {v.dtype} self: {self.dtype}"
            )

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
            idx_radial = ind_radial + np.arange(0, k_max, dtype=np.int)

            # include the normalization factor of angular part
            ang_nrms = self.angular_norms[idx_radial]
            radial = self._precomp["radial"][:, idx_radial]
            radial = radial / ang_nrms

            sgns = (1,) if ell == 0 else (1, -1)
            for _ in sgns:
                ang = self._precomp["ang"][:, ind_ang]
                ang_radial = np.expand_dims(ang[ang_idx], axis=1) * radial[r_idx]
                idx = ind + np.arange(0, k_max, dtype=np.int)
                x[mask] += ang_radial @ v[idx]
                ind += len(idx)
                ind_ang += 1

            ind_radial += len(idx_radial)

        x = x.T.reshape(-1, *self.sz)  # RCOPT

        return x

    def evaluate_t(self, v):
        """
        Evaluate coefficient in FB basis from those in standard 2D coordinate basis

        :param v: The coefficient array to be evaluated. The last dimensions
            must equal `self.sz`.
        :return: The evaluation of the coefficient array `v` in the dual basis
            of `basis`. This is an array of vectors whose last dimension equals
             `self.count` and whose first dimensions correspond to
             first dimensions of `v`.
        """

        if v.dtype != self.dtype:
            logger.warning(
                f"{self.__class__.__name__}::evaluate_t"
                f" Inconsistent dtypes v: {v.dtype} self: {self.dtype}"
            )

        v = v.T  # RCOPT

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
