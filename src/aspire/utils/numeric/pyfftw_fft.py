from threading import Lock

import numpy as np
import pyfftw

mutex = Lock()


class PyfftwFFT:

    # no fftshift and ifftshift from pyffw,use those from numpy
    fftshift = staticmethod(np.fft.fftshift)
    ifftshift = staticmethod(np.fft.ifftshift)

    @staticmethod
    def fft(a, axis=-1):
        mutex.acquire()

        try:
            a_ = pyfftw.empty_aligned(a.shape, dtype='complex128')
            b = pyfftw.empty_aligned(a.shape, dtype='complex128')
            cls = pyfftw.FFTW(a_, b, axes=(axis,), direction='FFTW_FORWARD')
            cls(a, b)
        finally:
            mutex.release()

        return b

    @staticmethod
    def fft2(a, axes=(0, 1)):
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
            cls = pyfftw.FFTW(a_, b, axes=axes, direction='FFTW_FORWARD')
            cls(a, b)
        finally:
            mutex.release()

        return b

    @staticmethod
    def fftn(a, axes):
        mutex.acquire()

        try:
            a_ = pyfftw.empty_aligned(a.shape, dtype='complex128')
            b = pyfftw.empty_aligned(a.shape, dtype='complex128')
            cls = pyfftw.FFTW(a_, b, axes=axes, direction='FFTW_FORWARD')
            cls(a, b)
        finally:
            mutex.release()

        return b

    @staticmethod
    def ifft(a, axis=-1):
        mutex.acquire()

        try:
            a_ = pyfftw.empty_aligned(a.shape, dtype='complex128')
            b = pyfftw.empty_aligned(a.shape, dtype='complex128')
            cls = pyfftw.FFTW(a_, b, axes=(axis,), direction='FFTW_BACKWARD')
            cls(a, b)
        finally:
            mutex.release()

        return b

    @staticmethod
    def ifft2(a, axes=(0, 1)):
        mutex.acquire()

        try:
            a_ = pyfftw.empty_aligned(a.shape, dtype='complex128')
            b = pyfftw.empty_aligned(a.shape, dtype='complex128')
            cls = pyfftw.FFTW(a_, b, axes=axes, direction='FFTW_BACKWARD')
            cls(a, b)
        finally:
            mutex.release()

        return b

    @staticmethod
    def ifftn(a, axes):
        mutex.acquire()

        try:
            a_ = pyfftw.empty_aligned(a.shape, dtype='complex128')
            b = pyfftw.empty_aligned(a.shape, dtype='complex128')
            cls = pyfftw.FFTW(a_, b, axes=axes, direction='FFTW_BACKWARD')
            cls(a, b)
        finally:
            mutex.release()

        return b
