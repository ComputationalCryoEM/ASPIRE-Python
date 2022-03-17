import logging

import numpy as np

from aspire.basis import Basis
from aspire.basis.basis_utils import num_besselj_zeros
from aspire.utils.matlab_compat import m_reshape

logger = logging.getLogger(__name__)


class FBBasis(Basis):
    """
    FBBasis is a mixin extension of Basis for subclasses that are expected to have
    methods specific to computing Fourier-Bessel expansion coefficients.
    """

    def _getfbzeros(self):
        """
        Generate zeros of Bessel functions
        """
        # get upper_bound of zeros of Bessel functions
        upper_bound = min(self.ell_max + 1, 2 * self.nres + 1)

        # List of number of zeros
        n = []
        # List of zero values (each entry is an ndarray; all of possibly different lengths)
        zeros = []

        # generate zeros of Bessel functions for each ell
        for ell in range(upper_bound):
            # for each ell, num_besselj_zeros returns the zeros of the
            # order ell Bessel function which are less than 2*pi*c*R = nres*pi/2,
            # the truncation rule for the Fourier-Bessel expansion
            _n, _zeros = num_besselj_zeros(
                ell + (self.ndim - 2) / 2, self.nres * np.pi / 2
            )
            if _n == 0:
                break
            else:
                n.append(_n)
                zeros.append(_zeros)

        #  get maximum number of ell
        self.ell_max = len(n) - 1

        #  set the maximum of k for each ell
        self.k_max = np.array(n, dtype=int)

        max_num_zeros = max(len(z) for z in zeros)
        for i, z in enumerate(zeros):
            zeros[i] = np.hstack(
                (z, np.zeros(max_num_zeros - len(z), dtype=self.dtype))
            )

        self.r0 = m_reshape(np.hstack(zeros), (-1, self.ell_max + 1)).astype(self.dtype)
