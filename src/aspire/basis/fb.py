import logging

import numpy as np

from aspire.basis.basis_utils import all_besselj_zeros

logger = logging.getLogger(__name__)


class FBBasisMixin(object):
    """
    FBBasisMixin is a mixin implementing methods specific to Fourier-Bessel expansions,
    to be inherited by Fourier-Bessel subclasses of Basis.
    """

    def _calc_k_max(self):
        """
        Generate zeros of Bessel functions
        """
        # get upper_bound of zeros of Bessel functions
        upper_bound = min(self.ell_max + 1, 2 * self.nres + 1)

        # List of number of zeros
        n = []
        # List of zero values (each entry is an ndarray; all of possibly different lengths)
        zeros = []

        for ell in range(upper_bound):
            # for each ell, num_besselj_zeros returns the zeros of the
            # order ell Bessel function which are less than 2*pi*c*R = nres*pi/2,
            # the truncation rule for the Fourier-Bessel expansion
            if self.ndim == 2:
                bessel_order = ell
            elif self.ndim == 3:
                bessel_order = ell + 1 / 2
            _n, _zeros = all_besselj_zeros(bessel_order, self.nres * np.pi / 2)
            if _n == 0:
                break
            else:
                n.append(_n)
                zeros.append(_zeros)

        #  get maximum number of ell
        self.ell_max = len(n) - 1

        #  set the maximum of k for each ell
        self.k_max = np.array(n, dtype=int)

        # set the zeros for each ell
        # this is a ragged list of 1d ndarrays, where the i'th element is of size self.k_max[i]
        self.r0 = zeros
