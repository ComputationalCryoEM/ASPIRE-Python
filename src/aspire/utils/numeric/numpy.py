from threading import Lock

import numpy as np
import pyfftw

mutex = Lock()


class Numpy:

    asnumpy = staticmethod(lambda x: x)

    @staticmethod
    def fft2(a, axes=(0, 1)):
        a_ = pyfftw.empty_aligned(a.shape, dtype='complex128')
        b = pyfftw.empty_aligned(a.shape, dtype='complex128')

        # This is called by ApplePicker unit test using ThreadPoolExecutor.
        #   I don't believe this pyfftw call is actually threadsafe.
        #   Holding mutex here, I have not been able to reproduce the spurious
        #   segmentation fault on Linux.
        #   This still allows threading in the other areas of invoking code,
        #   presumably the parts which are IO bound.
        mutex.acquire()

        try:
            cls = pyfftw.FFTW(a_, b, axes=axes, direction='FFTW_FORWARD')
            cls(a, b)
        finally:
            mutex.release()

        return b

    @staticmethod
    def ifft2(a, axes=(0, 1)):
        a_ = pyfftw.empty_aligned(a.shape, dtype='complex128')
        b = pyfftw.empty_aligned(a.shape, dtype='complex128')
        cls = pyfftw.FFTW(a_, b, axes=axes, direction='FFTW_BACKWARD')
        cls(a, b)
        return b

    def __getattr__(self, item):
        """
        Catch-all method to to allow a straight pass-through \
        of any attribute that is not supported above.
        """
        return getattr(np, item)
