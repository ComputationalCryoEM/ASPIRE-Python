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
