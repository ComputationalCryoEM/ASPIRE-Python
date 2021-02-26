import logging

import numpy as np
import pycuda.autoinit  # noqa: F401
import pycuda.driver as cuda  # noqa: F401
import pycuda.gpuarray as gpuarray  # noqa: F401
from cufinufft import cufinufft

from aspire.nufft import Plan
from aspire.utils import ensure

logger = logging.getLogger(__name__)


class CufinufftPlan(Plan):
    def __init__(self, sz, fourier_pts, epsilon=1e-15, ntransforms=1, **kwargs):
        """
        A plan for non-uniform FFT in 2D or 3D.

        :param sz: A tuple indicating the geometry of the signal
        :param fourier_pts: The points in Fourier space where the Fourier transform is to be calculated,
            arranged as a dimension-by-K array. These need to be in the range [-pi, pi] in each dimension.
        :param epsilon: The desired precision of the NUFFT
        :param ntransforms: Optional integer indicating if you would like to compute a batch of `ntransforms`
        transforms.  Implies vol_f.shape is (..., `ntransforms`). Defaults to 0 which disables batching.
        """

        # Passing "ntransforms" > 1 expects one large higher dimensional array later.
        self.ntransforms = ntransforms

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

        if not fourier_pts.flags.c_contiguous:
            logger.debug(
                "cufinufft has caught a non C_CONTIGUOUS array,"
                " `fourier_pts` will be copied to C_CONTIGUOUS."
            )
        self.fourier_pts = np.ascontiguousarray(
            np.mod(fourier_pts + np.pi, 2 * np.pi) - np.pi, dtype=self.dtype
        )

        self.num_pts = fourier_pts.shape[1]
        self.epsilon = max(epsilon, np.finfo(self.dtype).eps)

        self._transform_plan = cufinufft(
            2, self.sz, self.ntransforms, self.epsilon, -1, dtype=self.dtype
        )

        self.adjoint_opts = dict()
        if self.dtype is np.float64 and self.dim == 3 and self.epsilon < 1e3:
            # Note this is an algorithmic implementation dictated by shmem.
            logger.info(
                "Converting cufinufft gpu_method=1 from default of 2 for 3D1 transform,"
                f"to support computation in double precision with tol={self.epsilon}."
            )
            self.adjoint_opts["gpu_method"] = 1

        self._adjoint_plan = cufinufft(
            1,
            self.sz,
            self.ntransforms,
            self.epsilon,
            1,
            dtype=self.dtype,
            **self.adjoint_opts,
        )

        # Note, I store self.fourier_pts_gpu so the GPUArrray life
        #   is tied to instance, instead of this method.
        self.fourier_pts_gpu = gpuarray.to_gpu(self.fourier_pts)

        self._transform_plan.set_pts(*self.fourier_pts_gpu)
        self._adjoint_plan.set_pts(*self.fourier_pts_gpu)

    def transform(self, signal):
        """
        Compute the NUFFT transform using this plan instance.

        :param signal: Signal to be transformed. For a single transform,
        this should be a a 1, 2, or 3D array matching the plan `sz`.
        For a batch, signal should have shape `(*sz, ntransforms)`.

        :returns: Transformed signal of shape `num_pts` or
        `(ntransforms, num_pts)`.
        """

        if not (signal.dtype == self.dtype or signal.dtype == self.complex_dtype):
            logger.warning(
                "Incorrect dtypes passed to (a)nufft."
                " In the future this will be an error."
            )

        sig_shape = signal.shape
        res_shape = self.num_pts
        # Note, there is a corner case for ntransforms == 1.
        if self.ntransforms > 1 or (
            self.ntransforms == 1 and len(signal.shape) == self.dim + 1
        ):
            ensure(
                len(signal.shape) == self.dim + 1,
                f"For multiple transforms, {self.dim}D signal should be"
                f" a {self.ntransforms} element stack of {self.sz}.",
            )
            ensure(
                signal.shape[0] == self.ntransforms,
                "For multiple transforms, signal stack length"
                f" should match ntransforms {self.ntransforms}.",
            )

            sig_shape = signal.shape[1:]  # order...
            res_shape = (self.ntransforms, self.num_pts)

        ensure(
            sig_shape == self.sz,
            f"Signal frame to be transformed must have shape {self.sz}",
        )

        signal_gpu = gpuarray.to_gpu(
            np.ascontiguousarray(signal, dtype=self.complex_dtype)
        )

        result_gpu = gpuarray.GPUArray(res_shape, dtype=self.complex_dtype)

        self._transform_plan.execute(result_gpu, signal_gpu)

        result = result_gpu.get()

        return result

    def adjoint(self, signal):
        """
        Compute the NUFFT adjoint using this plan instance.

        :param signal: Signal to be transformed. For a single transform,
        this should be a a 1D array of len `num_pts`.
        For a batch, signal should have shape `(ntransforms, num_pts)`.

        :returns: Transformed signal `(sz)` or `(sz, ntransforms)`.
        """

        if not (signal.dtype == self.complex_dtype or signal.dtype == self.dtype):
            logger.warning(
                "Incorrect dtypes passed to (a)nufft."
                " In the future this will be an error."
            )

        res_shape = self.sz
        # Note, there is a corner case for ntransforms == 1.
        if self.ntransforms > 1 or (self.ntransforms == 1 and len(signal.shape) == 2):
            ensure(
                len(signal.shape) == 2,  # Stack and num_pts
                f"For multiple {self.dim}D adjoints, signal should be"
                f" a {self.ntransforms} element stack of {self.num_pts}.",
            )
            ensure(
                signal.shape[0] == self.ntransforms,
                "For multiple transforms, signal stack length"
                f" should match ntransforms {self.ntransforms}.",
            )
            res_shape = (self.ntransforms, *self.sz)

        signal_gpu = gpuarray.to_gpu(
            np.ascontiguousarray(signal, dtype=self.complex_dtype)
        )

        result_gpu = gpuarray.GPUArray(res_shape, dtype=self.complex_dtype)

        self._adjoint_plan.execute(signal_gpu, result_gpu)

        result = result_gpu.get()

        return result
