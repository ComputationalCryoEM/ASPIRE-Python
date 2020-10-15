import scipy as sp


class ScipyFFT:
    """
    Define a unified wrapper class for Scipy FFT functions

    To be consistent with Pyfftw, not all arguments are included.
    """
    @staticmethod
    def fft(x, axis=-1, workers=-1):
        return sp.fft.fft(x, axis=axis, workers=workers)

    @staticmethod
    def ifft(x, axis=-1, workers=-1):
        return sp.fft.ifft(x, axis=axis, workers=workers)

    @staticmethod
    def fft2(x, axes=(-2, -1), workers=-1):
        return sp.fft.fft2(x, axes=axes, workers=workers)

    @staticmethod
    def ifft2(x,  axes=(-2, -1), workers=-1):
        return sp.fft.ifft2(x, axes=axes, workers=workers)

    @staticmethod
    def fftn(x, axes=None, workers=-1):
        return sp.fft.fftn(x, axes=axes, workers=workers)

    @staticmethod
    def ifftn(x,  axes=None, workers=-1):
        return sp.fft.ifftn(x, axes=axes, workers=workers)

    @staticmethod
    def fftshift(x, axes=None):
        return sp.fft.fftshift(x, axes=axes)

    @staticmethod
    def ifftshift(x, axes=None):
        return sp.fft.ifftshift(x, axes=axes)

    @staticmethod
    def centered_ifft(x, axis=-1, workers=-1):
        x = sp.fft.ifftshift(x, axes=axis)
        x = sp.fft.ifft(x, axis=axis, workers=workers)
        x = sp.fft.fftshift(x, axes=axis)
        return x

    @staticmethod
    def centered_fft(x, axis=-1, workers=-1):
        x = sp.fft.ifftshift(x, axes=axis)
        x = sp.fft.fft(x, axis=axis, workers=workers)
        x = sp.fft.fftshift(x, axes=axis)
        return x

    @staticmethod
    def centered_ifft2(x, axes=(-2, -1), workers=-1):
        x = sp.fft.ifftshift(x, axes=axes)
        x = sp.fft.ifft2(x, axes=axes, workers=workers)
        x = sp.fft.fftshift(x, axes=axes)
        return x

    @staticmethod
    def centered_fft2(x, axes=(-2, -1), workers=-1):
        x = sp.fft.ifftshift(x, axes=axes)
        x = sp.fft.fft2(x, axes=axes, workers=workers)
        x = sp.fft.fftshift(x, axes=axes)
        return x

    @staticmethod
    def centered_ifftn(x, axes=None, workers=-1):
        x = sp.fft.ifftshift(x, axes=axes)
        x = sp.fft.ifftn(x, axes=axes, workers=workers)
        x = sp.fft.fftshift(x, axes=axes)
        return x

    @staticmethod
    def centered_fftn(x, axes=None, workers=-1):
        x = sp.fft.ifftshift(x, axes=axes)
        x = sp.fft.fftn(x, axes=axes, workers=workers)
        x = sp.fft.fftshift(x, axes=axes)
        return x
