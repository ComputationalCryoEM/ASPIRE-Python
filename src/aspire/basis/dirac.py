import logging
import numpy as np

from aspire.basis import Basis
from aspire.utils.matrix import roll_dim, unroll_dim
from aspire.utils.matlab_compat import m_flatten, m_reshape

logger = logging.getLogger(__name__)


class DiracBasis(Basis):
    """
    Define a derived class for Dirac basis
    """
    def __init__(self, sz, mask=None):
        """
        initialize an object for Dirac Basis
        :param sz: The size of the vectors for which to define the basis.
        :param mask: A boolean mask of size sz indicating which coordinates
            to include in the basis (default true(sz)).
        """
        if mask is None:
            mask = np.ones(sz)
            mask = mask > 0

        super().__init__(sz)

        self.mask = m_flatten(mask)
        self.count = np.sum(mask)
        self.sz_prod = self.nres**self.ndim

    def _build(self):
        """
        Build the internal data structure to Dirac basis
        """
        logger.info('Expanding object in a Dirac basis.')

    def evaluate(self, v):
        """
        Evaluate coefficients in standard coordinate basis from those in Dirac basis

        :param v: A coefficient vector (or an array of coefficient vectors) to
            be evaluated. The first dimension must equal `self.count`.
        :return: The evaluation of the coefficient vector(s) `v` for this basis.
            This is an array whose first dimensions equal `self.sz` and the remaining
            dimensions correspond to dimensions two and higher of `v`.
        """
        v, sz_roll = unroll_dim(v, 2)
        x = np.zeros(shape=tuple([self.sz_prod] + list(v.shape[1:])))
        x[self.mask, ...] = v
        x = m_reshape(x, self.sz + x.shape[1:])
        x = roll_dim(x, sz_roll)

        return x

    def evaluate_t(self, x):
        """
        Evaluate coefficient in Dirac basis from those in standard coordinate basis

        :param x: The coefficient array to be evaluated. The first dimensions
            must equal `self.sz`.
        :return: The evaluation of the coefficient array `v` in the dual basis
            of `basis`. This is an array of vectors whose first dimension equals
             `self.count` and whose remaining dimensions correspond to
             higher dimensions of `v`.
        """
        x, sz_roll = unroll_dim(x, self.ndim + 1)
        x = m_reshape(x, new_shape=tuple([self.sz_prod] + list(x.shape[self.ndim:])))
        v = np.zeros(shape=tuple([self.count] + list(x.shape[1:])))
        v = x[self.mask, ...]
        v = roll_dim(v, sz_roll)

        return v

    def expand(self, x):
        return self.evaluate_t(x)

    def expand_t(self, x):
        return self.evaluate_t(x)
