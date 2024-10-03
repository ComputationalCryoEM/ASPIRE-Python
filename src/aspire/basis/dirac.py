import logging

import numpy as np

from aspire.basis import Basis

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
            mask = np.full(size, True)
        if mask.shape != size:
            raise ValueError(f"Invalid mask size. Should match {size} or `None`.")
        # Ensure boolean mask
        self.mask = np.array(mask, dtype=bool)

        super().__init__(size, dtype=dtype)

    def _build(self):
        """Private method building basis internals."""
        self.count = np.count_nonzero(self.mask)

    def _evaluate(self, v):
        """
        Evaluate stack of standard coordinate coefficients from Dirac basis.

        :param v: Dirac basis coefficents. [..., self.count]
        :return:  Standard basis coefficients. [..., *self.sz]
        """

        # Initialize zeros array of standard basis size.
        x = np.zeros((v.shape[0], *self.sz), dtype=self.dtype)

        # Assign basis coefficient values
        x[..., self.mask] = v

        return x

    def expand(self, x):
        """
        See _evaluate.
        """
        return self.evaluate_t(x)

    def _evaluate_t(self, x):
        """
        Evaluate stack of Dirac basis coefficients from standard basis.

        :param x:  Standard basis coefficients. [..., *self.sz]
        :return: Dirac basis coefficents. [..., self.count]
        """

        # Initialize zeros array of dirac basis (mask) count.
        v = np.zeros((x.shape[0], self.count), dtype=self.dtype)

        # Assign basis coefficient values
        v = x[..., self.mask]

        return v


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
