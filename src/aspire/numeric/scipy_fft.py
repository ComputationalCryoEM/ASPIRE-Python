import scipy as sp

from aspire.numeric.base_fft import FFT


class ScipyFFT(FFT):
    """
    Define a unified wrapper class for Scipy FFT functions

    To be consistent with Pyfftw, not all arguments are included.
    """

    def fft(self, x, axis=-1, workers=-1):
        return sp.fft.fft(x, axis=axis, workers=workers)

    def ifft(self, x, axis=-1, workers=-1):
        return sp.fft.ifft(x, axis=axis, workers=workers)

    def fft2(self, x, axes=(-2, -1), workers=-1):
        return sp.fft.fft2(x, axes=axes, workers=workers)

    def ifft2(self, x, axes=(-2, -1), workers=-1):
        return sp.fft.ifft2(x, axes=axes, workers=workers)

    def fftn(self, x, axes=None, workers=-1):
        return sp.fft.fftn(x, axes=axes, workers=workers)

    def ifftn(self, x, axes=None, workers=-1):
        return sp.fft.ifftn(x, axes=axes, workers=workers)

    def fftshift(self, x, axes=None):
        return sp.fft.fftshift(x, axes=axes)

    def ifftshift(self, x, axes=None):
        return sp.fft.ifftshift(x, axes=axes)

    def dct(self, x, **kwargs):
        return sp.fft.dct(x, **kwargs)

    def idct(self, x, **kwargs):
        return sp.fft.idct(x, **kwargs)

    def rfftfreq(self, x, **kwargs):
        return sp.fft.rfftfreq(x, **kwargs)

    def irfft(self, x, **kwargs):
        return sp.fft.irfft(x, **kwargs)

    def rfft(self, x, **kwargs):
        return sp.fft.rfft(x, **kwargs)

    def irfft2(self, x, **kwargs):
        return sp.fft.irfft2(x, **kwargs)

    def rfft2(self, x, **kwargs):
        return sp.fft.rfft2(x, **kwargs)
