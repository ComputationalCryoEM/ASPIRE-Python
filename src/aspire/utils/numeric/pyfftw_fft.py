import pyfftw.interfaces.scipy_fftpack as pyfft


class PyfftwFFT:

    fft = staticmethod(pyfft.fft)
    ifft = staticmethod(pyfft.ifft)
    fft2 = staticmethod(pyfft.fft2)
    ifft2 = staticmethod(pyfft.ifft2)
    fftn = staticmethod(pyfft.fftn)
    ifftn = staticmethod(pyfft.ifftn)

    fftshift = staticmethod(pyfft.fftshift)
    ifftshift = staticmethod(pyfft.ifftshift)
