import cupy as cp


class CupyFFT:

    fft = staticmethod(cp.fft.fft)
    ifft = staticmethod(cp.fft.ifft)
    fft2 = staticmethod(cp.fft.fft2)
    ifft2 = staticmethod(cp.fft.ifft2)
    fftn = staticmethod(cp.fft.fftn)
    ifftn = staticmethod(cp.fft.ifftn)

    fftshift = staticmethod(cp.fft.fftshift)
    ifftshift = staticmethod(cp.fft.ifftshift)
