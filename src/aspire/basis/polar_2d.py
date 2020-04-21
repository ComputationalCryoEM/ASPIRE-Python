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
    Define a derived class for polar Fourier expansion for 2D images
    """

    def __init__(self, size, nrad=None, ntheta=None):
        """
        Initialize an object for the 2D polar Fourier basis class

        :param size: The size of the vectors for which to define the basis.
            Currently only square images are supported.
        :param nrad: The number of points in the radial dimension.
        :param ntheta: The number of points in the angular dimension.
        """

        ndim = len(size)
        ensure(ndim == 2, 'Only two-dimensional basis functions are supported.')
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
        Build the internal data structure to 2D polar Fourier basis
        """
        logger.info('Expanding 2D image in a polar Fourier basis')

        self.count = self.nrad * self.ntheta
        self.sz_prod = self.sz[0] * self.sz[1]

        # precompute the basis functions in 2D grids
        self.freqs = self._precomp()

    def _precomp(self):
        """
        Precomute the basis functions on a polar Fourier grid
        """
        omega0 = 2 * np.pi / (2 * self.nrad - 1)
        dtheta = 2 * np.pi / self.ntheta

        # Note: The size of ntheta can be reduced by half if only evaluate_t function is needed as
        # in the Matlab implementation, np.zeros((2, self.nrad * self.ntheta//2)).
        freqs = np.zeros((2, self.nrad * self.ntheta))
        for i in range(self.ntheta):
            freqs[0, i * self.nrad: (i + 1) * self.nrad] = np.arange(self.nrad) * np.sin(i * dtheta)
            freqs[1, i * self.nrad: (i + 1) * self.nrad] = np.arange(self.nrad) * np.cos(i * dtheta)

        freqs *= omega0
        return freqs

    def evaluate_t(self, x):
        """
        Evaluate coefficient in polar Fourier basis from those in standard 2D coordinate basis

        :param x: The coefficient array in the standard 2D coordinate basis to be
            evaluated. The first two dimensions must equal `self.sz`.
        :return v: The evaluation of the coefficient array `v` in the polar Fourier basis.
            This is an array of vectors whose first two dimensions are `self.nrad` and
            `self.ntheta` and whose remaining dimensions correspond to higher dimensions of `x`.
        """
        # ensure the first two dimensions with size of self.sz
        x, sz_roll = unroll_dim(x, self.ndim + 1)
        x = m_reshape(x, (self.sz[0], self.sz[1], -1))
        nimgs = x.shape[2]

        # finufftpy require it to be aligned in fortran order
        half_size = self.nrad * self.ntheta // 2
        pf = np.empty((half_size, nimgs), dtype='complex128', order='F')
        finufftpy.nufft2d2many(self.freqs[0, 0:half_size], self.freqs[1, 0:half_size], pf, 1, 1e-15, x)
        pf = m_reshape(pf, (self.nrad, self.ntheta // 2, nimgs))
        v = np.concatenate((pf, pf.conj()), axis=1).copy()

        # return v coefficients with the first dimension of self.count
        v = roll_dim(v, sz_roll)
        return v

    def expand(self, x):
        """
        Obtain coefficients in polar Fourier basis from those in standard 2D coordinate basis

        This is the same function to `evaluate_t` for a consistent implementation
        with other expanding methods.

        :param x: The coefficient array in the standard 2D coordinate basis to be
            evaluated. The first two dimensions must equal `self.sz`.
        :return v: The evaluation of the coefficient array `v` in the polar Fourier basis.
            This is an array of vectors whose first two dimensions are `self.nrad` and
            `self.ntheta` and whose remaining dimensions correspond to higher dimensions of `x`.
        """
        return self.evaluate_t(x)
