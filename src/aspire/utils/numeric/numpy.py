import numpy as np
import pyfftw

from threading import Lock

mutex = Lock()

class Numpy:

    asnumpy = staticmethod(lambda x: x)

    @staticmethod
    def fft2(a, axes=(0, 1)):
        a_ = pyfftw.empty_aligned(a.shape, dtype='complex128')
        b = pyfftw.empty_aligned(a.shape, dtype='complex128')
        # GBW, I don't believe this pyfftw is actually threadsafe.
        #   Holding mutex here, I have not been able to reproduce the spurious
        #   segmentation fault on Linux.
        #   This still allows threading in the other areas of invoking code,
        #   presumably the parts which are IO bound...
        mutex.acquire()
        try:
            cls = pyfftw.FFTW(a_, b, axes=axes, direction='FFTW_FORWARD')
            cls(a, b)
        finally:
            mutex.release()
        return b

    @staticmethod
    def ifft2(a, axes=(0, 1)):
        b = pyfftw.empty_aligned(a.shape, dtype='complex128')
        cls = pyfftw.FFTW(pyfftw.empty_aligned(a.shape, dtype='complex128'), b, axes=axes, direction='FFTW_BACKWARD')
        cls(a, b)
        return b

    def __getattr__(self, item):
        """
        Catch-all method to to allow a straight pass-through of any attribute that is not supported above.
        """
        return getattr(np, item)
