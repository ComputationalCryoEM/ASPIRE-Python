import logging

import numpy as np

from aspire.basis import Basis
from aspire.image import Image

logger = logging.getLogger(__name__)


class FLEBasis2D(Basis):
    """
    FLE Basis.

    https://arxiv.org/pdf/2207.13674.pdf
    """

    def __init__(self, size, bandlimit=None, epsilon=None, dtype=np.float32):
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
        super().__init__(self, size, ell_max=None, dtype=self.dtype)

    def _build(self):
        # Heuristic for max iterations
        maxitr = 1 + int(3 * np.log2(self.nres))
        numsparse = 32
        if eps >= 1e-10:
            numsparse = 22
            maxitr = 1 + int(2 * np.log2(L))
        if eps >= 1e-7:
            numsparse = 16
            maxitr = 1 + int(np.log2(L))
        if eps >= 1e-4:
            numsparse = 8
            maxitr = 1 + int(np.log2(L)) // 2
        self.maxitr = maxitr
        self.numsparse = numsparse

    def _precomp(self):

        # Regular Fourier-Bessel bandlimit (equivalent to pi*R**2)
        self.num_basis_functions = int(L**2 * np.pi / 4)

        # Number of radial nodes
        # (Lemma 4.1)
        # compute max {2.4 * self.nres , Log2 ( 1 / epsilon) }
        Q = int(np.ceil(2.4 * self.nres))
        tmp = 1 / (np.sqrt(np.pi))
        for q in range(1, Q + 1):
            tmp = tmp / q * (np.sqrt(np.pi) * L / 4)
            if tmp <= self.epsilon:
                num_radial_nodes = int(max(q, np.log2(1 / self.epsilon)))
                break
        num_radial_nodes = max(num_radial_nodes, int(np.ceil(np.log2(1 / eps))))

        # Number of angular nodes
        # (Lemma 4.2)
        # compute max {7.08 * self.nres, Log2(1/epsilon) + Log2(self.nres**2) }
        S = int(max(7.08*self.nres, -np.log2(epsilon) + 2*np.log2(self.nres)))
        num_angular_nodes = S
        for s in range(int(lmd1 + ndmax) + 1, S + 1):
            tmp = L**2 * ((lmd1 + ndmax) / s)**s
            if tmp <= self.epsilon:
                num_angular_nodes = int(max(int(s), np.log2(1/self.epsilon)))
                break

        # must be even
        if num_angular_nodes % 2 == 1:
            num_angular_nodes += 1

    def _lap_eig_disk(self):
        """
        Compute the eigenvalues of the Laplacian operator on a disk with Dirichlet boundary conditions.
        """
        
        
        # Maximum bandlimit
        # (Section 4.1)
        # Can remove frequencies above this threshold based on the fact that
        # there should not be more basis functions than pixels contained in the
        # unit disk inscribed on the image
        if self.bandlimit:
            pass
        
        
