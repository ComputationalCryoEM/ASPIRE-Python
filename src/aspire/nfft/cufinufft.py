from ctypes import c_int

import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from cufinufft import cufinufft

from aspire.nfft import Plan
from aspire.utils import ensure


class cuFINufftPlan(Plan):
    def __init__(self, sz, fourier_pts, epsilon=1e-15, many=0, **kwargs):
        # Passing "many" expects one large higher dimensional array.
        #   Set some housekeeping variables so we can discern how to handle the dims later.
        self.many = False
        self.ntransforms = 1
        if many != 0:
            self.many = True
            self.ntransforms = many

        # Basic dtype passthough.
        dtype = fourier_pts.dtype
        if dtype == np.float64 or dtype == np.complex128:
            self.dtype = np.float64
            self.complex_dtype = np.complex128
        elif dtype == np.float32 or dtype == np.complex64:
            self.dtype = np.float32
            self.complex_dtype = np.complex64
        else:
            raise RuntimeError("Unsupported dtype encountered")

        self.sz = sz
        self.dim = len(sz)
        # TODO: Things get messed up unless we ensure a 'C' ordering here - investigate why
        self.fourier_pts = np.asarray(np.mod(fourier_pts + np.pi, 2 * np.pi) - np.pi, order='C', dtype=self.dtype)
        self.num_pts = fourier_pts.shape[1]
        self.epsilon = max(epsilon, np.finfo(self.dtype).eps)

        self._transform_plan = cufinufft(2, self.sz, -1, self.epsilon, ntransforms=self.ntransforms,
                                         dtype=self.dtype)

        self._adjoint_plan = cufinufft(1, self.sz, 1, self.epsilon, ntransforms=self.ntransforms,
                                       dtype=self.dtype)

    def __del__(self):
        for plan in [self._transform_plan, self._adjoint_plan]:
            plan.destroy()

    def transform(self, signal):

        assert signal.dtype == self.dtype or signal.dtype == self.complex_dtype

        sig_shape = signal.shape
        if self.many:
            sig_shape = signal.shape[:-1]         # order...
        ensure(sig_shape == self.sz, f'Signal to be transformed must have shape {self.sz}')

        signal_gpu = gpuarray.to_gpu(signal.astype(self.complex_dtype, copy=False, order='F'))

        # This ordering situation is a little strange, but it works.
        result_gpu = gpuarray.GPUArray((self.ntransforms, self.num_pts), dtype=self.complex_dtype, order='C')

        fourier_pts_gpu = gpuarray.to_gpu(self.fourier_pts.astype(self.dtype))
        self._transform_plan.set_nu_pts(self.num_pts, *fourier_pts_gpu)

        # move to init or something, check handles 1,2,3 d (I think it unrolls okay)
        self._transform_plan.execute(result_gpu, signal_gpu)

        result = result_gpu.get()

        if not self.many:
            result = result[0]

        return result

    def adjoint(self, signal):

        assert signal.dtype == self.complex_dtype or signal.dtype == self.dtype
        if self.dim == 3 and self.dtype == np.float64:
            raise TypeError('Currently the 3d1 sub-problem method is singles only.')

        signal_gpu = gpuarray.to_gpu(signal.astype(self.complex_dtype, copy=False, order='C'))

        # This ordering situation is a little strange, but it works.
        result_gpu = gpuarray.GPUArray((*self.sz, self.ntransforms), dtype=self.complex_dtype, order='F')

        fourier_pts_gpu = gpuarray.to_gpu(self.fourier_pts)
        self._adjoint_plan.set_nu_pts(self.num_pts, *fourier_pts_gpu)

        self._adjoint_plan.execute(signal_gpu, result_gpu)

        result = result_gpu.get()

        if not self.many:
            result = result[...,0]

        return result
