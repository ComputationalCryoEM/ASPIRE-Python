import mkl_fft
import numpy as np

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
