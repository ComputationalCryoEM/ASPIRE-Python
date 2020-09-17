import cupy as cp


class Cupy:

    fft = staticmethod(cp.fft.fft)
    ifft = staticmethod(cp.fft.ifft)
    fft2 = staticmethod(cp.fft.fft2)
    ifft2 = staticmethod(cp.fft.ifft2)
    fftn = staticmethod(cp.fft.fftn)
    ifftn = staticmethod(cp.fft.ifftn)

    fftshift = staticmethod(cp.fft.fftshift)
    ifftshift = staticmethod(cp.fft.ifftshift)

    def __getattr__(self, item):
        """
        Catch-all method to to allow a straight pass-through of any attribute that is not supported above.
        """
        return getattr(cp, item)
