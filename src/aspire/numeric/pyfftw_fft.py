import os
from threading import Lock

import pyfftw
import pyfftw.interfaces.scipy_fftpack as scipy_fft

from aspire.numeric.base_fft import FFT
from aspire.utils.types import complex_type

mutex = Lock()

_cpu_count = os.cpu_count()


def _workers(workers):
    if workers in (None, 0):
        raise ValueError("Workers must be specified")

    if workers < 0:
        # Borrow the idea from scipy for negative values of workers.
        # When workers is -1, we use all available threads.
        # Otherwise, (-workers-1) will be saved for other tasks.
        if workers >= -_cpu_count:
            workers += 1 + _cpu_count
        else:
            raise ValueError(
                f"Workers value out of range; got {workers}, "
                f"must not be less than {-_cpu_count}"
            )

    return workers


class PyfftwFFT(FFT):
    """
    Define a unified wrapper class for PyFFT functions

    To be consistent with Scipy FFT, not all arguments are included.
    """

    def fft(self, a, axis=-1, workers=-1):
        mutex.acquire()

        comp_type = complex_type(a.dtype)
        try:
            a_ = pyfftw.empty_aligned(a.shape, dtype=comp_type)
            b = pyfftw.empty_aligned(a.shape, dtype=comp_type)
            cls = pyfftw.FFTW(
                a_, b, axes=(axis,), direction="FFTW_FORWARD", threads=_workers(workers)
            )
            cls(a, b)
        finally:
            mutex.release()

        return b

    def ifft(self, a, axis=-1, workers=-1):
        mutex.acquire()

        comp_type = a.dtype
        try:
            a_ = pyfftw.empty_aligned(a.shape, dtype=comp_type)
            b = pyfftw.empty_aligned(a.shape, dtype=comp_type)
            cls = pyfftw.FFTW(
                a_,
                b,
                axes=(axis,),
                direction="FFTW_BACKWARD",
                threads=_workers(workers),
            )
            cls(a, b)
        finally:
            mutex.release()

        return b

    def fft2(self, a, axes=(-2, -1), workers=-1):
        # This is called by ApplePicker unit test using ThreadPoolExecutor.
        #   I don't believe this pyfftw call is actually threadsafe.
        #   Holding mutex here, I have not been able to reproduce the spurious
        #   segmentation fault on Linux.
        #   This still allows threading in the other areas of invoking code,
        #   presumably the parts which are IO bound.
        mutex.acquire()

        comp_type = complex_type(a.dtype)
        try:
            a_ = pyfftw.empty_aligned(a.shape, dtype=comp_type)
            b = pyfftw.empty_aligned(a.shape, dtype=comp_type)
            cls = pyfftw.FFTW(
                a_, b, axes=axes, direction="FFTW_FORWARD", threads=_workers(workers)
            )
            cls(a, b)
        finally:
            mutex.release()

        return b

    def ifft2(self, a, axes=(-2, -1), workers=-1):
        mutex.acquire()

        comp_type = a.dtype
        try:
            a_ = pyfftw.empty_aligned(a.shape, dtype=comp_type)
            b = pyfftw.empty_aligned(a.shape, dtype=comp_type)
            cls = pyfftw.FFTW(
                a_, b, axes=axes, direction="FFTW_BACKWARD", threads=_workers(workers)
            )
            cls(a, b)
        finally:
            mutex.release()

        return b

    def fftn(self, a, axes=None, workers=-1):
        mutex.acquire()

        comp_type = complex_type(a.dtype)
        try:
            a_ = pyfftw.empty_aligned(a.shape, dtype=comp_type)
            b = pyfftw.empty_aligned(a.shape, dtype=comp_type)
            cls = pyfftw.FFTW(
                a_, b, axes=axes, direction="FFTW_FORWARD", threads=_workers(workers)
            )
            cls(a, b)
        finally:
            mutex.release()

        return b

    def ifftn(self, a, axes=None, workers=-1):
        mutex.acquire()

        comp_type = a.dtype
        try:
            a_ = pyfftw.empty_aligned(a.shape, dtype=comp_type)
            b = pyfftw.empty_aligned(a.shape, dtype=comp_type)
            cls = pyfftw.FFTW(
                a_, b, axes=axes, direction="FFTW_BACKWARD", threads=_workers(workers)
            )
            cls(a, b)
        finally:
            mutex.release()

        return b

    def fftshift(self, a, axes=None):
        return scipy_fft.fftshift(a, axes=axes)

    def ifftshift(self, a, axes=None):
        return scipy_fft.ifftshift(a, axes=axes)
