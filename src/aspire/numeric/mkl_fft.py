import mkl_fft
import numpy as np
import scipy as sp

from aspire.numeric.base_fft import FFT


class MKLFFT(FFT):
    """
    Define a unified wrapper class for MKL FFT functions

    To be consistent with Pyfftw, not all arguments are included.
    """

    # Note MKL does use the "workers" argument
    def fft(self, x, axis=-1, workers=-1):
        return mkl_fft.fft(
            x,
            axis=axis,
        )

    def ifft(self, x, axis=-1, workers=-1):
        return mkl_fft.ifft(
            x,
            axis=axis,
        )

    def fft2(self, x, axes=(-2, -1), workers=-1):
        return mkl_fft.fft2(
            x,
            axes=axes,
        )

    def ifft2(self, x, axes=(-2, -1), workers=-1):
        return mkl_fft.ifft2(
            x,
            axes=axes,
        )

    def fftn(self, x, axes=None, workers=-1):
        return mkl_fft.fftn(
            x,
            axes=axes,
        )

    def ifftn(self, x, axes=None, workers=-1):
        return mkl_fft.ifftn(
            x,
            axes=axes,
        )

    def fftshift(self, x, axes=None):
        # N/A in mkl_fft, use np
        return np.fft.fftshift(x, axes=axes)

    def ifftshift(self, x, axes=None):
        # N/A in mkl_fft, use np
        return np.fft.ifftshift(x, axes=axes)

    def rfft(self, x, **kwargs):
        return mkl_fft._numpy_fft.rfft(x, **kwargs)

    def irfft(self, x, **kwargs):
        return mkl_fft._numpy_fft.irfft(x, **kwargs)

    def rfft2(self, x, **kwargs):
        return mkl_fft._numpy_fft.rfft2(x, **kwargs)

    def irfft2(self, x, **kwargs):
        return mkl_fft._numpy_fft.irfft2(x, **kwargs)

    # These are not currently exposed in mkl_fft,
    #   fall back to scipy.
    def dct(self, x, **kwargs):
        return sp.fft.dct(x, **kwargs)

    def idct(self, x, **kwargs):
        return sp.fft.idct(x, **kwargs)

    def rfftfreq(self, x, **kwargs):
        return sp.fft.rfftfreq(x, **kwargs)
