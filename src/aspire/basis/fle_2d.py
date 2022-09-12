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
        if isinstance(size, int):
            size = (size, size)
        ndim = len(size)
        assert ndim == 2, "Only two-dimensional basis functions are supported."
        assert len(set(size)) == 1, "Only square domains are supported"

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

        # Compute basis functions
        self._lap_eig_disk()

        # Some important constants
        self.smallest_lambda = np.min(self.bessel_zeros)
        self.greatest_lambda = np.max(self.bessel_zeros)
        # TODO: explain
        self.ndmax = np.max(2 * np.abs(self.ells) - (self.ells < 0))

        # radial and angular nodes for fast Chebyshev interpolation
        self._compute_chebyshev_nodes()

    def _compute_chebyshev_nodes(self):
        """
        Compute the number of radial and angular nodes for fast Chebyshev interpolation
        """

        # Number of radial nodes
        # (Lemma 4.1)
        # compute max {2.4 * self.nres , Log2 ( 1 / epsilon) }
        Q = int(np.ceil(2.4 * self.nres))
        num_radial_nodes = Q
        tmp = 1 / (np.sqrt(np.pi))
        for q in range(1, Q + 1):
            tmp = tmp / q * (np.sqrt(np.pi) * self.nres / 4)
            if tmp <= self.epsilon:
                num_radial_nodes = int(max(q, np.log2(1 / self.epsilon)))
                break
        self.num_radial_nodes = max(
            num_radial_nodes, int(np.ceil(np.log2(1 / self.epsilon)))
        )

        # Number of angular nodes
        # (Lemma 4.2)
        # compute max {7.08 * self.nres, Log2(1/epsilon) + Log2(self.nres**2) }

        S = int(max(7.08 * self.nres, -np.log2(self.epsilon) + 2 * np.log2(self.nres)))
        num_angular_nodes = S
        for s in range(int(self.greatest_lambda + self.ndmax) + 1, S + 1):
            tmp = self.nres**2 * ((self.greatest_lambda + self.ndmax) / s) ** s
            if tmp <= self.epsilon:
                num_angular_nodes = int(max(int(s), np.log2(1 / self.epsilon)))
                break

        # must be even
        if num_angular_nodes % 2 == 1:
            num_angular_nodes += 1

        self.num_angular_nodes = num_angular_nodes

    def _lap_eig_disk(self):
        """
        Compute the eigenvalues of the Laplacian operator on a disk with Dirichlet boundary conditions.
        """
        # max number of Bessel function orders being considered
        max_ell = int(3 * np.sqrt(self.max_basis_functions))
        # max number of zeros per Bessel function (number of frequencies per bessel)
        max_k = int(2 * np.sqrt(self.max_basis_functions))

        # preallocate containers for roots
        # 0 frequency plus pos and negative frequencies for each bessel function
        # num functions per frequency
        num_ells = 1 + 2 * max_ell
        self.ells = np.zeros((num_ells, max_k), dtype=int, order="F")
        self.ks = np.zeros((num_ells, max_k), dtype=int, order="F")
        self.bessel_zeros = np.ones((num_ells, max_k), dtype=np.float64) * np.Inf

        # keep track of which order Bessel function we're on
        self.ells[0, :] = 0
        # bessel_roots[0, m] is the m'th zero of J_0
        self.bessel_zeros[0, :] = besselj_zeros(0, max_k)
        # table of values of which zero of J_0 we are finding
        self.ks[0, :] = np.arange(max_k) + 1

        # add roots of J_ell for ell>0 twice with +k and -k (frequencies)
        # iterate over Bessel function order
        for ell in range(1, max_ell + 1):
            self.ells[2 * ell - 1, :] = -ell
            self.ks[2 * ell - 1, :] = np.arange(max_k) + 1

            self.bessel_zeros[2 * ell - 1, :max_k] = besselj_zeros(ell, max_k)

            self.ells[2 * ell, :] = ell
            self.ks[2 * ell, :] = self.ks[2 * ell - 1, :]
            self.bessel_zeros[2 * ell, :] = self.bessel_zeros[2 * ell - 1, :]

        ### bessel_zeros

        # [ R_0_1, R_0_2, R_0_3 ... R_0_maxk ]
        # [ R_1_1, R_1_2, R_1_3 ... R_1_maxk ]
        # [ R_1_1, R_1_2, R_1_3 ... R_1_maxk ]
        # [ R_2_1, R_2_2, R_2_3 ... R_2_maxk ]
        # [ R_2_1, R_2_2, R_2_3 ... R_2_maxk ]
        # ... ...
        # [ R_numells_1,R_numells_2,R_numells_3 ... R_numells_maxk ]
        # [ R_numells_1,R_numells_2,R_numells_3 ... R_numells_maxk ]

        ### ells

        # [ 0, 0, 0, ... 0 ] (max_k)
        # [-1,-1,-1, ...-1 ]
        # [ 1, 1, 1, ... 1 ]
        # [-2,-2,-2, ...-2 ]
        # [ 2, 2, 2, ... 2 ]
        # ... ...
        # [-num_ells,-num_ells,-num_ells...-num_ells ]
        # [ num_ells, num_ells, num_ells... num_ells ]

        ### ks

        # [1, 2, 3, ... max_k ]
        # [1, 2, 3, ... max_k ]
        # ...
        # [1, 2, 3, ... max_k ]

        # Reshape the arrays and order by the size of the Bessel function zeros
        self._flatten_and_sort_bessel_zeros()

        # Apply threshold criterion to throw out some basis functions
        # Grab final number of basis functions for this Basis
        self.count = self._threshold_basis_functions()

    def _flatten_and_sort_bessel_zeros(self):
        """
        Reshapes arrays self.ells, self.ks, and self.bessel_zeros
        """
        # flatten list of zeros, ells and ks:
        self.ells = self.ells.flatten()
        self.ks = self.ks.flatten()
        self.bessel_zeros = self.bessel_zeros.flatten()

        ### TODO: Better way of doing the next two sections
        ### (Specifically ordering the neg and pos integers in the correct way)
        # sort by size of zeros
        idx = np.argsort(self.bessel_zeros)
        self.ells = self.ells[idx]
        self.ks = self.ks[idx]
        # sort complex conjugate pairs: -n first, +n second
        idx = np.arange(self.max_basis_functions + 1)
        for i in range(self.max_basis_functions + 1):
            if self.ells[i] >= 0:
                continue
            if np.abs(self.bessel_zeros[i] - self.bessel_zeros[i + 1]) < 1e-14:
                continue
            idx[i - 1] = i
            idx[i] = i - 1

        self.ells = self.ells[idx]
        self.ks = self.ks[idx]
        self.bessel_zeros = self.bessel_zeros[idx]

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
            for i in range(len(self.bessel_zeros)):
                if (
                    self.bessel_zeros[_final_num_basis_functions] / (np.pi)
                    >= (self.bandlimit - 1) // 2
                ):
                    _final_num_basis_functions -= 1

        # potentially subtract one to keep complex conjugate pairs
        if self.ells[_final_num_basis_functions - 1] < 0:
            _final_num_basis_functions -= 1

        # discard zeros above the threshold
        self.ells = self.ells[:_final_num_basis_functions]
        self.ks = self.ks[:_final_num_basis_functions]
        self.bessel_zeros = self.bessel_zeros[:_final_num_basis_functions]

        return _final_num_basis_functions

    def _evaluate(self, coeffs):
        """
        Placeholder.

        Evaluate FLE coefficients and return in standard 2D Cartesian coordinates.

        :param v: A coefficient vector (or an array of coefficient vectors) to
            be evaluated. The last dimension must be equal to `self.count`
        """
        return np.zeros((coeffs.shape[0], self.nres, self.nres))

    def _evaluate_t(self, imgs):
        """
        Placeholder.

        Evaluate 2D Cartesian image(s) and return the corresponding FLE coefficients.

        :param imgs: The array to be evaluated. The last dimensions
            must equal `self.sz`
        """

        return np.zeros((imgs.shape[0],) + (self.count,))
