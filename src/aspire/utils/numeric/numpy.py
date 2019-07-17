import numpy as np
import pyfftw


class Numpy:

    asnumpy = staticmethod(lambda x: x)

    @staticmethod
    def fft2(a, axes=(0, 1)):
        b = pyfftw.empty_aligned(a.shape, dtype='complex128')
        cls = pyfftw.FFTW(pyfftw.empty_aligned(a.shape, dtype='complex128'), b, axes=axes, direction='FFTW_FORWARD')
        cls(a, b)
        return b

    @staticmethod
    def ifft2(a, axes=(0, 1)):
        b = pyfftw.empty_aligned(a.shape, dtype='complex128')
        cls = pyfftw.FFTW(pyfftw.empty_aligned(a.shape, dtype='complex128'), b, axes=axes, direction='FFTW_BACKWARD')
        cls(a, b)
        return b

    def __getattr__(self, item):
        """
        Catch-all method to to allow a straight pass-through of any attribute that is not supported above.
        """
        return getattr(np, item)
