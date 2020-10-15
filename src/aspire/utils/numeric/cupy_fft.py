import cupy as cp


class CupyFFT:
    """
    Define a unified wrapper class for Cupy FFT functions

    To be consistent with Scipy and Pyfftw, not all arguments are included.
    """
    @staticmethod
    def fft(x, axis=-1, workers=-1):
        return cp.fft.fft(x, axis=axis)

    @staticmethod
    def ifft(x, axis=-1, workers=-1):
        return cp.fft.ifft(x, axis=axis)

    @staticmethod
    def fft2(x, axes=(-2, -1), workers=-1):
        return cp.fft.fft2(x, axes=axes)

    @staticmethod
    def ifft2(x,  axes=(-2, -1), workers=-1):
        return cp.fft.ifft2(x, axes=axes)

    @staticmethod
    def fftn(x, axes=None, workers=-1):
        return cp.fft.fftn(x, axes=axes)

    @staticmethod
    def ifftn(x,  axes=None, workers=-1):
        return cp.fft.ifft2(x, axes=axes)

    @staticmethod
    def fftshift(x, axes=None):
        return cp.fft.fftshift(x, axes=axes)

    @staticmethod
    def ifftshift(x, axes=None):
        return cp.fft.ifftshift(x, axes=axes)

    @staticmethod
    def centered_ifft(x, axis=-1, workers=-1):
        x = cp.fft.ifftshift(x, axes=axis)
        x = cp.fft.ifft(x, axis=axis)
        x = cp.fft.fftshift(x, axes=axis)
        return x

    @staticmethod
    def centered_fft(x, axis=-1, workers=-1):
        x = cp.fft.ifftshift(x, axes=axis)
        x = cp.fft.fft(x, axis=axis)
        x = cp.fft.fftshift(x, axes=axis)
        return x

    @staticmethod
    def centered_ifft2(x, axes=(-2, -1), workers=-1):
        x = cp.fft.ifftshift(x, axes=axes)
        x = cp.fft.ifft2(x, axes=axes)
        x = cp.fft.fftshift(x, axes=axes)
        return x

    @staticmethod
    def centered_fft2(x, axes=(-2, -1), workers=-1):
        x = cp.fft.ifftshift(x, axes=axes)
        x = cp.fft.fft2(x, axes=axes)
        x = cp.fft.fftshift(x, axes=axes)
        return x

    @staticmethod
    def centered_ifftn(x, axes=None, workers=-1):
        x = cp.fft.ifftshift(x, axes=axes)
        x = cp.fft.ifftn(x, axes=axes)
        x = cp.fft.fftshift(x, axes=axes)
        return x

    @staticmethod
    def centered_fftn(x, axes=None, workers=-1):
        x = cp.fft.ifftshift(x, axes=axes)
        x = cp.fft.fftn(x, axes=axes)
        x = cp.fft.fftshift(x, axes=axes)
        return x
