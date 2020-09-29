import scipy.fft as spfft


class ScipyFFT:

    fft = staticmethod(spfft.fft)
    ifft = staticmethod(spfft.ifft)
    fft2 = staticmethod(spfft.fft2)
    ifft2 = staticmethod(spfft.ifft2)
    fftn = staticmethod(spfft.fftn)
    ifftn = staticmethod(spfft.ifftn)

    fftshift = staticmethod(spfft.fftshift)
    ifftshift = staticmethod(spfft.ifftshift)
