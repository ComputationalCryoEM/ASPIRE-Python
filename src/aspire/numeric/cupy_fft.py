import functools

import cupy as cp
import cupyx.scipy.fft as cufft
import numpy as np

from aspire.numeric.base_fft import FFT


# This improves the flexibility of our FFT wrappers by allowing for
# incremental code changes and testing.
def _preserve_host(func):
    """
    Method decorator that returns a numpy/cupy array result when
    passed a numpy/cupy array input respectively.

    At the time of writing this wrapper will also upcast cupy FFT
    operations to doubles as the precision in singles can cause
    accuracy issues.
    """

    @functools.wraps(func)  # Pass metadata (eg name and doctrings) from `func`
    def wrapper(self, x, *args, **kwargs):

        # CuPy's single precision FFT appears to be too inaccurate for
        # many of our unit tests, so the signal is upcast and recast
        # on return.
        _singles = False
        if x.dtype == np.float32:
            _singles = True
            x = x.astype(np.float64)
        elif x.dtype == np.complex64:
            _singles = True
            x = x.astype(np.complex128)

        _host = False
        if not isinstance(x, cp.ndarray):
            _host = True
            x = cp.asarray(x)

        res = func(self, x, *args, **kwargs)

        if _host:
            res = res.get()

        # Recast if needed.
        if _singles and res.dtype == np.float64:
            res = res.astype(np.float32)
        elif _singles and res.dtype == np.complex128:
            res = res.astype(np.complex64)

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

    def rfftfreq(self, n, **kwargs):
        return cufft.rfftfreq(n, **kwargs)

    @_preserve_host
    def irfft(self, x, **kwargs):
        return cufft.irfft(x, **kwargs)

    @_preserve_host
    def rfft(self, x, **kwargs):
        return cufft.rfft(x, **kwargs)
