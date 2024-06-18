import functools

import cupy as cp
import cupyx.scipy.fft as cufft

from aspire.numeric.base_fft import FFT


def _preserve_host(func):
    """
    Method decorator that returns a numpy/cupy array result when passed a numpy/cupy array input.

    This improves the flexibility of our FFT wrappers by allowing for incremental code changes.
    """

    @functools.wraps(func)  # Pass metadata (eg name and doctrings) from `func`
    def wrapper(self, x, *args, **kwargs):

        _host = False
        if not isinstance(x, cp.ndarray):
            _host = True
            x = cp.asarray(x)

        res = func(self, x, *args, **kwargs)

        if _host:
            res = res.get()

        return res

    return wrapper


class CupyFFT(FFT):
    """
    Define a unified wrapper class for Cupy FFT functions

    To be consistent with Scipy and Pyfftw, not all arguments are included.
    """

    @_preserve_host
    def fft(self, x, axis=-1, workers=-1):
        return cp.fft.fft(x, axis=axis)

    @_preserve_host
    def ifft(self, x, axis=-1, workers=-1):
        return cp.fft.ifft(x, axis=axis)

    @_preserve_host
    def fft2(self, x, axes=(-2, -1), workers=-1):
        return cp.fft.fft2(x, axes=axes)

    @_preserve_host
    def ifft2(self, x, axes=(-2, -1), workers=-1):
        return cp.fft.ifft2(x, axes=axes)

    @_preserve_host
    def fftn(self, x, axes=None, workers=-1):
        return cp.fft.fftn(x, axes=axes)

    @_preserve_host
    def ifftn(self, x, axes=None, workers=-1):
        return cp.fft.ifftn(x, axes=axes)

    @_preserve_host
    def fftshift(self, x, axes=None):
        return cp.fft.fftshift(x, axes=axes)

    @_preserve_host
    def ifftshift(self, x, axes=None):
        return cp.fft.ifftshift(x, axes=axes)

    @_preserve_host
    def dct(self, x, **kwargs):
        return cufft.dct(x, **kwargs)

    @_preserve_host
    def idct(self, x, **kwargs):
        return cufft.idct(x, **kwargs)
