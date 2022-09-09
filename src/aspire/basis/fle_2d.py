import logging

import numpy as np

from aspire.basis import FBBasisMixin, SteerableBasis2D
from aspire.basis.basis_utils import besselj_zeros
from aspire.image import Image

logger = logging.getLogger(__name__)


class FLEBasis2D(SteerableBasis2D, FBBasisMixin):
    """
    FLE Basis.

    https://arxiv.org/pdf/2207.13674.pdf
    """

    def __init__(self, size, bandlimit=None, epsilon=1e-10, dtype=np.float32):
        """
        :param size: The size of the vectors for which to define the FLE basis.
                 Currently only square images are supported.
        :param bandlimit: Maximum frequency band for computing basis functions. Note that the
            `ell_max` of other Basis objects is computed *from* the bandlimit for the FLE basis.
        :param epsilon: Relative precision between FLE fast method and dense matrix multiplication.
        """
        self.size = size
        self.bandlimit = bandlimit
        self.epsilon = epsilon
        self.dtype = dtype
        # Basis.__init__()
        super().__init__(size, ell_max=None, dtype=self.dtype)

    def _build(self):
        # get upper bound of zeros, ells, and ks of Bessel functions
        # self._calc_k_max()
        # self.count = self.k_max[0] + sum(2*self.k_max[1:])

        # Heuristic for max iterations
        maxitr = 1 + int(3 * np.log2(self.nres))
        numsparse = 32
        if self.epsilon >= 1e-10:
            numsparse = 22
            maxitr = 1 + int(2 * np.log2(self.nres))
        if self.epsilon >= 1e-7:
            numsparse = 16
            maxitr = 1 + int(np.log2(self.nres))
        if self.epsilon >= 1e-4:
            numsparse = 8
            maxitr = 1 + int(np.log2(self.nres)) // 2
        self.maxitr = maxitr
        self.numsparse = numsparse

        self._precomp()

    def _precomp(self):

        # Regular Fourier-Bessel bandlimit (equivalent to pi*R**2)
        self.max_basis_functions = int(self.nres**2 * np.pi / 4)

        self._lap_eig_disk()

        # Number of radial nodes
        # (Lemma 4.1)
        # compute max {2.4 * self.nres , Log2 ( 1 / epsilon) }
        Q = int(np.ceil(2.4 * self.nres))
        tmp = 1 / (np.sqrt(np.pi))
        for q in range(1, Q + 1):
            tmp = tmp / q * (np.sqrt(np.pi) * self.nres / 4)
            if tmp <= self.epsilon:
                num_radial_nodes = int(max(q, np.log2(1 / self.epsilon)))
                break
        num_radial_nodes = max(
            num_radial_nodes, int(np.ceil(np.log2(1 / self.epsilon)))
        )

        # Number of angular nodes
        # (Lemma 4.2)
        # compute max {7.08 * self.nres, Log2(1/epsilon) + Log2(self.nres**2) }

    #        S = int(max(7.08*self.nres, -np.log2(epsilon) + 2*np.log2(self.nres)))
    #        num_angular_nodes = S
    #        for s in range(int(lmd1 + ndmax) + 1, S + 1):
    #            tmp = self.nres**2 * ((lmd1 + ndmax) / s)**s
    #            if tmp <= self.epsilon:
    #                num_angular_nodes = int(max(int(s), np.log2(1/self.epsilon)))
    #                break

    # must be even
    #        if num_angular_nodes % 2 == 1:
    #            num_angular_nodes += 1

    def _lap_eig_disk(self):
        """
        Compute the eigenvalues of the Laplacian operator on a disk with Dirichlet boundary conditions.
        """
        # max number of Bessel function orders being considered
        nc = int(3 * np.sqrt(self.max_basis_functions))
        # max number of zeros per Bessel function (number of frequencies per bessel)
        nd = int(2 * np.sqrt(self.max_basis_functions))

        # preallocate containers for roots
        # 0 frequency plus pos and negative frequencies for each bessel function
        # num functions per frequency
        nn = 1 + 2 * nc
        self.ns = np.zeros((nn, nd), dtype=int, order="F")
        self.ks = np.zeros((nn, nd), dtype=int, order="F")
        self.bessel_roots = np.ones((nn, nd), dtype=np.float64) * np.Inf

        ### bessel_roots

        # [ R_0_1, R_0_2, R_0_3 ... R_0_nd ]
        # [ R_1_1, R_1_2, R_1_3 ... R_1_nd ]
        # [ R_1_1, R_1_2, R_1_3 ... R_1_nd ]
        # [ R_2_1, R_2_2, R_2_3 ... R_2_nd ]
        # [ R_2_1, R_2_2, R_2_3 ... R_2_nd ]
        # ... ...
        # [ R_nn_1,R_nn_2,R_nn_3 ... R_nn_nd ]
        # [ R_nn_1,R_nn_2,R_nn_3 ... R_nn_nd ]

        ### ns

        # [ 0, 0, 0, ... 0 ] (nn)
        # [-1,-1,-1, ...-1 ]
        # [ 1, 1, 1, ... 1 ]
        # [-2,-2,-2, ...-2 ]
        # [ 2, 2, 2, ... 2 ]
        # ... ...
        # [-nn,-nn,-nn...-nn ]
        # [ nn, nn, nn... nn ]

        ### ks

        # [1, 2, 3, ... nd ]
        # [1, 2, 3, ... nd ]
        # ...
        # [1, 2, 3, ... nd ]

        # Initially compute the max Bessel function zeros that will be needed
        # and organize the n's (Bessel function orders) and k's (frequencies)
        # into tables
        self._populate_bessel_roots()

        # Reshape the arrays and order by the size of the Bessel function zeros
        self._flatten_and_sort_bessel_roots()

        # Apply threshold criterion to throw out some basis functions
        # Grab final number of basis functions for this Basis
        self.num_basis_functions = self._threshold_basis_functions()

    def _populate_bessel_roots(self):
        """
        Populates self.ns, self.ks, and self.bessel_roots.
        """
        # keep track of which order Bessel function we're on
        self.ns[0, :] = 0
        # bessel_roots[0, m] is the m'th zero of J_0
        self.bessel_roots[0, :] = besselj_zeros(0, nd)
        # table of values of which zero of J_0 we are finding
        self.ks[0, :] = np.arange(nd) + 1

        # add roots of J_n for n>0 twice with +k and -k (frequencies)
        # iterate over Bessel function order
        for n in range(1, nc + 1):
            self.ns[2 * n - 1, :] = -n
            self.ks[2 * n - 1, :] = np.arange(nd) + 1

            self.bessel_roots[2 * n - 1, :nd] = besselj_zeros(n, nd)

            self.ns[2 * n, :] = n
            self.ks[2 * n, :] = self.ks[2 * n - 1, :]
            self.bessel_roots[2 * n, :] = self.bessel_roots[2 * n - 1, :]

    def _flatten_and_sort_bessel_roots(self):
        """
        Reshapes arrays self.ns, self.ks, and self.bessel_roots
        """
        # flatten list of zeros, ns and ks:
        self.ns = self.ns.flatten()
        self.ks = self.ks.flatten()
        self.bessel_roots = self.bessel_roots.flatten()

        ### TODO: Better way of doing the next two sections
        ### (Specifically ordering the neg and pos integers in the correct way)
        # sort by size of zeros
        idx = np.argsort(self.bessel_roots)
        self.ns = self.ns[idx]
        self.ks = self.ks[idx]
        # sort complex conjugate pairs: -n first, +n second
        idx = np.arange(self.max_basis_functions + 1)
        for i in range(self.max_basis_functions + 1):
            if self.ns[i] >= 0:
                continue
            if np.abs(self.bessel_roots[i] - self.bessel_roots[i + 1]) < 1e-14:
                continue
            idx[i - 1] = i
            idx[i] = i - 1

        self.ns = self.ns[idx]
        self.ks = self.ks[idx]
        self.bessel_roots = self.bessel_roots[idx]

    def _threshold_basis_functions(self):
        """
        Implements the bandlimit threshold which caps the number of basis functions
        that are actually required.
        :return: The final overall number of basis functions to be used.
        """
        # Maximum bandlimit
        # (Section 4.1)
        # Can remove frequencies above this threshold based on the fact that
        # there should not be more basis functions than pixels contained in the
        # unit disk inscribed on the image
        _final_num_basis_functions = self.max_basis_functions
        if self.bandlimit:
            for i in range(len(self.bessel_roots)):
                if (
                    self.bessel_roots[_final_num_basis_functions] / (np.pi)
                    >= (self.bandlimit - 1) // 2
                ):
                    _final_num_basis_functions -= 1

        # potentially subtract one to keep complex conjugate pairs
        if self.ns[_final_num_basis_functions - 1] < 0:
            _final_num_basis_functions -= 1

        # discard zeros above the threshold
        self.ns = self.ns[:_final_num_basis_functions]
        self.ks = self.ks[:_final_num_basis_functions]
        self.bessel_roots = self.bessel_roots[:_final_num_basis_functions]

        return _final_num_basis_functions
