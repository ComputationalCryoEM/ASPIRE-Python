import logging

import numpy as np

from aspire.basis import Basis
from aspire.numeric import xp

logger = logging.getLogger(__name__)


class DiracBasis(Basis):
    """
    Dirac basis in 1D.

    Supports subclassing higher dimensions.
    """

    dim = 1

    def __init__(self, size, mask=None, dtype=np.float32):
        """
        Initialize Dirac basis.

        :param size: The shape defining the basis.  May be a tuple
            or an integer, in which case, a uniformly sized basis is assumed.
        :param mask: A boolean mask matching `size` indicating which
            coordinates to include in the basis. Default `None`
            implies all via `np.full((size,)*dimension, True)`.
        :return: DiracBasis2D instance.
        """

        # Size
        if isinstance(size, int):
            size = (size,) * self.dim

        # Masking
        if mask is None:
            mask = xp.full(size, True)
        if mask.shape != size:
            raise ValueError(f"Invalid mask size. Should match {size} or `None`.")
        # Ensure boolean mask
        self.mask = xp.asarray(mask, dtype=bool)

        super().__init__(size, dtype=dtype)

    def _build(self):
        """Private method building basis internals."""
        self.count = int(xp.count_nonzero(self.mask))

    def _evaluate(self, v):
        """
        Evaluate stack of standard coordinate coefficients from Dirac basis.

        Given numpy/cupy array, returns numpy/cupy respectively.

        :param v: Dirac basis coefficents. [..., self.count]
        :return:  Standard basis coefficients. [..., *self.sz]
        """
        _zeros = xp.zeros  # Default to xp array
        mask = self.mask
        if not isinstance(v, xp.ndarray):
            # Do not use cupy when `v` _and_ xp are not both cupy ndarray
            # Avoids having to handle when cupy is not installed
            #
            # v is host, x and mask should be on host as well.
            _zeros = np.zeros
            mask = xp.asnumpy(self.mask)

        # Initialize zeros array of standard basis size.
        x = _zeros((v.shape[0], *self.sz), dtype=self.dtype)

        # Assign basis coefficient values
        x[..., mask] = v

        return x

    def expand(self, x):
        """
        See _evaluate.
        """
        return self.evaluate_t(x)

    def _evaluate_t(self, x):
        """
        Evaluate stack of Dirac basis coefficients from standard basis.

        Given numpy/cupy array, returns numpy/cupy respectively.

        :param x:  Standard basis coefficients. [..., *self.sz]
        :return: Dirac basis coefficents. [..., self.count]
        """
        mask = self.mask
        if not isinstance(x, xp.ndarray):
            # Do not use cupy when `x` _and_ xp are not both cupy ndarray
            # Avoids having to handle when cupy is not installed
            mask = xp.asnumpy(self.mask)

        # Applying the mask should flatten mask.ndim axes
        return x[..., mask]


class DiracBasis2D(DiracBasis):
    """
    Dirac basis in 2D.

    See `DiracBasis` documentation.
    """

    dim = 2


class DiracBasis3D(DiracBasis):
    """
    Dirac basis in 3D.

    See `DiracBasis` documentation.
    """

    dim = 3
