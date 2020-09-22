from scipy import fftpack


class ScipyFFT:

    fft = staticmethod(fftpack.fft)
    ifft = staticmethod(fftpack.ifft)
    fft2 = staticmethod(fftpack.fft2)
    ifft2 = staticmethod(fftpack.ifft2)
    fftn = staticmethod(fftpack.fftn)
    ifftn = staticmethod(fftpack.ifftn)

    fftshift = staticmethod(fftpack.fftshift)
    ifftshift = staticmethod(fftpack.ifftshift)
