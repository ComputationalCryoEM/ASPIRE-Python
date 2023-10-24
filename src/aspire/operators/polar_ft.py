import logging

import numpy as np

from aspire.image import Image
from aspire.nufft import nufft
from aspire.utils import complex_type

logger = logging.getLogger(__name__)


class PolarFT:
    """
    Define a derived class for polar Fourier representation for 2D images
    """

    def __init__(self, size, nrad=None, ntheta=None, dtype=np.float32):
        """
        Initialize an object for the polar Fourier transform class. `PolarFT` expects that
        images are real and uses only half of the `ntheta` values.

        :param size: The shape of the vectors for which to define the transform.
            May be a 2-tuple or an integer, in which case a square basis is assumed.
            Currently only square images are supported.
        :param nrad: The number of points in the radial dimension. Default is resolution // 2.
        :param ntheta: The number of points in the angular dimension. Default is 8 * nrad.
        :param dtype: dtype used to compute a polar frequency grid for evaluating the transform..
        """
        if isinstance(size, int):
            size = (size, size)
        ndim = len(size)
        assert ndim == 2, "Only two-dimensional grids are supported."
        assert len(set(size)) == 1, "Only square domains are supported."

        self.ndim = ndim
        self.sz = size
        self.nrad = nrad
        self.ntheta = ntheta
        self.dtype = dtype

        self._build()

        # this basis has complex coefficients
        self.coefficient_dtype = complex_type(self.dtype)

    def _build(self):
        """
        Build the internal data structure to 2D polar Fourier grid
        """
        logger.info("Represent 2D image in a polar Fourier grid")

        if self.nrad is None:
            self.nrad = self.sz[0] // 2

        if self.ntheta is None:
            # try to use the same number as Fast FB basis
            self.ntheta = 8 * self.nrad

        if self.ntheta % 2 == 1:
            msg = "Only even values for ntheta are supported."
            logger.error(msg)
            raise NotImplementedError(msg)

        self.count = self.nrad * (self.ntheta // 2)
        self._sz_prod = self.sz[0] * self.sz[1]

        # precompute the basis functions in 2D grids
        self.freqs = self._precomp()

    def _precomp(self):
        """
        Precompute the polar Fourier grid.
        """
        omega0 = 2 * np.pi / (2 * self.nrad - 1)
        dtheta = 2 * np.pi / self.ntheta

        # only need half size of ntheta
        freqs = np.zeros((2, self.ntheta // 2, self.nrad), dtype=self.dtype)
        for i in range(self.ntheta // 2):
            freqs[0, i] = np.sin(i * dtheta)
            freqs[1, i] = np.cos(i * dtheta)

        freqs *= omega0 * np.arange(self.nrad)

        return freqs.reshape(2, -1)

    def transform(self, x):
        """
        Evaluate coefficient in polar Fourier grid from those in standard 2D coordinate basis

        :param x: The Image instance representing coefficient array in the
            standard 2D coordinate basis to be evaluated.
        :return: The evaluation of the coefficient array `x` in the polar
            Fourier grid. This is an array of vectors whose first dimension
            corresponds to `x.shape[0]`, and last dimension equals `self.count`.
        """
        if x.dtype != self.dtype:
            raise TypeError(
                f"{self.__class__.__name__}.transform"
                f" Inconsistent dtypes x: {x.dtype} self: {self.dtype}"
            )

        if not isinstance(x, Image):
            raise TypeError(
                f"{self.__class__.__name__}.transform"
                f" passed numpy array instead of {Image}."
            )
        else:
            x = x.asnumpy()

        # Flatten stack
        stack_shape = x.shape[: -self.ndim]
        x = x.reshape(-1, *x.shape[-self.ndim :])

        # We expect the Image `x` to be real in order to take advantage of the conjugate
        # symmetry of the Fourier transform of a real valued image.
        if not np.isreal(x).all():
            raise TypeError(
                f"The Image `x` must be real valued. Found dtype {x.dtype}."
            )

        resolution = x.shape[-1]

        pf = nufft(x, self.freqs) / resolution**2

        return pf.reshape(*stack_shape, self.ntheta // 2, self.nrad)

    @staticmethod
    def half_to_full(pf):
        """
        Use the conjugate symmetry of pf to construct the full polar Fourier transform
        over all rays in [0, 360).

        :param pf: The precomputed half polar Fourier transform
            with shape (*stack_shape, ntheta//2, nrad)
        :return: The full polar Fourier transform with shape (*stack_shape, ntheta, nrad)
        """

        return np.concatenate((pf, np.conj(pf)), axis=-2)
