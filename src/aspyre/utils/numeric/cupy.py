import cupy as cp


class Cupy:
    fft2 = staticmethod(cp.fft.fft2)
    ifft2 = staticmethod(cp.fft.ifft2)

    def __getattr__(self, item):
        """
        Catch-all method to to allow a straight pass-through of any attribute that is not supported above.
        """
        return getattr(cp, item)
