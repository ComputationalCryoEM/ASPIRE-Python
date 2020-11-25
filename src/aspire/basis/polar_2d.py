import logging

import numpy as np

from aspire.basis import Basis
from aspire.image import Image
from aspire.nufft import anufft, nufft
from aspire.utils import ensure, real_type

logger = logging.getLogger(__name__)


class PolarBasis2D(Basis):
    """
    Define a derived class for polar Fourier representation for 2D images
    """

    def __init__(self, size, nrad=None, ntheta=None, dtype=np.float32):
        """
        Initialize an object for the 2D polar Fourier grid class

        :param size: The shape of the vectors for which to define the grid.
            Currently only square images are supported.
        :param nrad: The number of points in the radial dimension.
        :param ntheta: The number of points in the angular dimension.
        """

        ndim = len(size)
        ensure(ndim == 2, "Only two-dimensional grids are supported.")
        ensure(len(set(size)) == 1, "Only square domains are supported.")

        self.nrad = nrad
        if nrad is None:
            self.nrad = self.nres // 2

        self.ntheta = ntheta
        if ntheta is None:
            # try to use the same number as Fast FB basis
            self.ntheta = 8 * self.nrad

        super().__init__(size, dtype=dtype)

    def _build(self):
        """
        Build the internal data structure to 2D polar Fourier grid
        """
        logger.info("Represent 2D image in a polar Fourier grid")

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

        # only need half size of ntheta
        freqs = np.zeros((2, self.nrad * self.ntheta // 2), dtype=self.dtype)
        for i in range(self.ntheta // 2):
            freqs[0, i * self.nrad : (i + 1) * self.nrad] = np.arange(
                self.nrad
            ) * np.cos(i * dtheta)
            freqs[1, i * self.nrad : (i + 1) * self.nrad] = np.arange(
                self.nrad
            ) * np.sin(i * dtheta)

        freqs *= omega0
        return freqs

    def evaluate(self, v):
        """
        Evaluate coefficients in standard 2D coordinate basis from those in polar Fourier basis

        :param v: A coefficient vector (or an array of coefficient vectors)
            in polar Fourier basis to be evaluated. The last dimension must equal to
            `self.count`.
        :return x: Image instance in standard 2D coordinate basis with
            resolution of `self.sz`.
        """
        if self.dtype != real_type(v.dtype):
            msg = (
                f"Input data type, {v.dtype}, is not consistent with"
                f" type defined in the class {self.dtype}."
            )
            logger.error(msg)
            raise TypeError(msg)

        v = v.reshape(-1, self.ntheta, self.nrad)

        nimgs = v.shape[0]

        half_size = self.ntheta // 2

        v = v[:, :half_size, :] + v[:, half_size:, :].conj()

        v = v.reshape(nimgs, self.nrad * half_size)

        x = anufft(v, self.freqs, self.sz, real=True)

        return Image(x)

    def evaluate_t(self, x):
        """
        Evaluate coefficient in polar Fourier grid from those in standard 2D coordinate basis

        :param x: The Image instance representing coefficient array in the
        standard 2D coordinate basis to be evaluated.
        :return v: The evaluation of the coefficient array `v` in the polar
        Fourier grid. This is an array of vectors whose first dimension
        corresponds to x.n_images, and last dimension equals `self.count`.
        """

        assert isinstance(x, Image)

        if self.dtype != x.dtype:
            msg = (
                f"Input data type, {x.dtype}, is not consistent with"
                f" type defined in the class {self.dtype}."
            )
            logger.error(msg)
            raise TypeError(msg)

        nimgs = x.n_images

        half_size = self.ntheta // 2

        pf = nufft(x.asnumpy(), self.freqs)

        pf = pf.reshape((nimgs, self.nrad, half_size))
        v = np.concatenate((pf, pf.conj()), axis=1)

        # return v coefficients with the last dimension size of self.count
        v = v.reshape(nimgs, -1)
        return v
