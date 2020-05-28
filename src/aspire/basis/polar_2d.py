import logging
import numpy as np
import finufftpy

from aspire.utils import ensure
from aspire.utils.matrix import roll_dim, unroll_dim
from aspire.utils.matlab_compat import m_reshape
from aspire.basis import Basis

logger = logging.getLogger(__name__)


class PolarBasis2D(Basis):
    """
    Define a derived class for polar Fourier representation for 2D images
    """

    def __init__(self, size, nrad=None, ntheta=None):
        """
        Initialize an object for the 2D polar Fourier grid class

        :param size: The shape of the vectors for which to define the grid.
            Currently only square images are supported.
        :param nrad: The number of points in the radial dimension.
        :param ntheta: The number of points in the angular dimension.
        """

        ndim = len(size)
        ensure(ndim == 2, 'Only two-dimensional grids are supported.')
        ensure(len(set(size)) == 1, 'Only square domains are supported.')

        self.nrad = nrad
        if nrad is None:
            self.nrad = self.nres // 2

        self.ntheta = ntheta
        if ntheta is None:
            # try to use the same number as Fast FB basis
            self.ntheta = 8 * self.nrad

        super().__init__(size)

    def _build(self):
        """
        Build the internal data structure to 2D polar Fourier grid
        """
        logger.info('Represent 2D image in a polar Fourier grid')

        self.count = self.nrad * self.ntheta
        self._sz_prod = self.sz[0] * self.sz[1]

        # precompute the basis functions in 2D grids
        self.freqs = self._precomp()

    def _precomp(self):
        """
        Precomute the polar Fourier grid
        """
        omega0 = 2 * np.pi / (2 * self.nrad - 1)
        dtheta = 2 * np.pi / self.ntheta

        freqs = np.zeros((2, self.nrad * self.ntheta))
        for i in range(self.ntheta):
            freqs[0, i * self.nrad: (i + 1) * self.nrad] = np.arange(self.nrad) * np.sin(i * dtheta)
            freqs[1, i * self.nrad: (i + 1) * self.nrad] = np.arange(self.nrad) * np.cos(i * dtheta)

        freqs *= omega0
        return freqs

    def evaluate(self, v):
        """
        Evaluate coefficients in standard 2D coordinate basis from those in polar Fourier basis

        :param v: A coefficient vector (or an array of coefficient vectors)
            in polar Fourier basis to be evaluated. The first dimension must equal to
            `self.count`.
        :return x: The evaluation of the coefficient vector(s) `x` in standard 2D
            coordinate basis. This is an array whose first two dimensions equal `self.sz`
            and the remaining dimensions correspond to dimensions two and higher of `v`.
        """
        v, sz_roll = unroll_dim(v, 2)
        nimgs = v.shape[1]

        # finufftpy require it to be aligned in fortran order
        x = np.empty((self._sz_prod, nimgs), dtype='complex128', order='F')
        finufftpy.nufft2d1many(self.freqs[0], self.freqs[1], v, 1, 1e-15,
                               self.sz[0], self.sz[1], x)
        x = m_reshape(x, (self.sz[0], self.sz[1], nimgs))
        # return coefficients whose first two dimensions equal to self.sz
        x = roll_dim(x, sz_roll)

        return x

    def evaluate_t(self, x):
        """
        Evaluate coefficient in polar Fourier grid from those in standard 2D coordinate basis

        :param x: The coefficient array in the standard 2D coordinate basis to be
            evaluated. The first two dimensions must equal `self.sz`.
        :return v: The evaluation of the coefficient array `v` in the polar Fourier grid.
            This is an array of vectors whose first dimension is `self.count` and
            whose remaining dimensions correspond to higher dimensions of `x`.
        """
        # ensure the first two dimensions with size of self.sz
        x, sz_roll = unroll_dim(x, self.ndim + 1)
        nimgs = x.shape[2]

        # finufftpy require it to be aligned in fortran order
        half_size = self.nrad * self.ntheta // 2
        pf = np.empty((half_size, nimgs), dtype='complex128', order='F')
        finufftpy.nufft2d2many(self.freqs[0, :half_size], self.freqs[1, :half_size], pf, 1, 1e-15, x)
        pf = m_reshape(pf, (self.nrad, self.ntheta // 2, nimgs))
        v = np.concatenate((pf, pf.conj()), axis=1)

        # return v coefficients with the first dimension size of self.count
        v = m_reshape(v, (self.nrad * self.ntheta, nimgs))
        v = roll_dim(v, sz_roll)

        return v
