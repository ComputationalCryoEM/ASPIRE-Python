from threading import Lock
import os
import pyfftw
import pyfftw.interfaces.scipy_fftpack as pyfft

mutex = Lock()


class PyfftwFFT:
    """
    Define a unified wrapper class for PyFFT functions

    To be consistent with Scipy FFT, not all arguments are included.
    """
    @staticmethod
    def fft(a, axis=-1, workers=-1):
        mutex.acquire()

        try:
            a_ = pyfftw.empty_aligned(a.shape, dtype='complex128')
            b = pyfftw.empty_aligned(a.shape, dtype='complex128')
            if workers == -1:
                threads = os.cpu_count()
            else:
                threads = workers
            cls = pyfftw.FFTW(a_, b, axes=(axis,), direction='FFTW_FORWARD',
                              threads=threads)
            cls(a, b)
        finally:
            mutex.release()

        return b

    @staticmethod
    def ifft(a, axis=-1, workers=-1):
        mutex.acquire()

        try:
            a_ = pyfftw.empty_aligned(a.shape, dtype='complex128')
            b = pyfftw.empty_aligned(a.shape, dtype='complex128')
            if workers == -1:
                threads = os.cpu_count()
            else:
                threads = workers
            cls = pyfftw.FFTW(a_, b, axes=(axis,), direction='FFTW_BACKWARD',
                              threads=threads)
            cls(a, b)
        finally:
            mutex.release()

        return b

    @staticmethod
    def fft2(a, axes=(-2, -1), workers=-1):
        # This is called by ApplePicker unit test using ThreadPoolExecutor.
        #   I don't believe this pyfftw call is actually threadsafe.
        #   Holding mutex here, I have not been able to reproduce the spurious
        #   segmentation fault on Linux.
        #   This still allows threading in the other areas of invoking code,
        #   presumably the parts which are IO bound.
        mutex.acquire()

        try:
            a_ = pyfftw.empty_aligned(a.shape, dtype='complex128')
            b = pyfftw.empty_aligned(a.shape, dtype='complex128')
            if workers == -1:
                threads = os.cpu_count()
            else:
                threads = workers
            cls = pyfftw.FFTW(a_, b, axes=axes, direction='FFTW_FORWARD',
                              threads=threads)
            cls(a, b)
        finally:
            mutex.release()

        return b

    @staticmethod
    def ifft2(a, axes=(-2, -1), workers=-1):
        mutex.acquire()

        try:
            a_ = pyfftw.empty_aligned(a.shape, dtype='complex128')
            b = pyfftw.empty_aligned(a.shape, dtype='complex128')
            if workers == -1:
                threads = os.cpu_count()
            else:
                threads = workers
            cls = pyfftw.FFTW(a_, b, axes=axes, direction='FFTW_BACKWARD',
                              threads=threads)
            cls(a, b)
        finally:
            mutex.release()

        return b

    @staticmethod
    def fftn(a, axes=None, workers=-1):
        mutex.acquire()

        try:
            a_ = pyfftw.empty_aligned(a.shape, dtype='complex128')
            b = pyfftw.empty_aligned(a.shape, dtype='complex128')
            if workers == -1:
                threads = os.cpu_count()
            else:
                threads = workers
            cls = pyfftw.FFTW(a_, b, axes=axes, direction='FFTW_FORWARD',
                              threads=threads)
            cls(a, b)
        finally:
            mutex.release()

        return b

    @staticmethod
    def ifftn(a, axes=None, workers=-1):
        mutex.acquire()

        try:
            a_ = pyfftw.empty_aligned(a.shape, dtype='complex128')
            b = pyfftw.empty_aligned(a.shape, dtype='complex128')
            if workers == -1:
                threads = os.cpu_count()
            else:
                threads = workers
            cls = pyfftw.FFTW(a_, b, axes=axes, direction='FFTW_BACKWARD',
                              threads=threads)
            cls(a, b)
        finally:
            mutex.release()

        return b

    @staticmethod
    def fftshift(a, axes=None):
        return pyfft.fftshift(a, axes=axes)

    @staticmethod
    def ifftshift(a, axes=None):
        return pyfft.ifftshift(a, axes=axes)
