import scipy as sp


class ScipyFFT:

    fft = staticmethod(sp.fftpack.fft)
    ifft = staticmethod(sp.fftpack.ifft)
    fft2 = staticmethod(sp.fftpack.fft2)
    ifft2 = staticmethod(sp.fftpack.ifft2)
    fftn = staticmethod(sp.fftpack.fftn)
    ifftn = staticmethod(sp.fftpack.ifftn)

    fftshift = staticmethod(sp.fftpack.fftshift)
    ifftshift = staticmethod(sp.fftpack.ifftshift)
