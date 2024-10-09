import logging

import cupy as cp
import numpy as np
from cufinufft import Plan as cufPlan

from aspire.nufft import Plan
from aspire.utils import complex_type

logger = logging.getLogger(__name__)


class CufinufftPlan(Plan):
    def __init__(self, sz, fourier_pts, epsilon=1e-8, ntransforms=1, **kwargs):
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

        # Workaround cufinufft A100 singles issue
        # ASPIRE-Python/703
        # Cast to doubles.
        self._original_dtype = fourier_pts.dtype
        fourier_pts = fourier_pts.astype(np.float64, copy=False)

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
        self.fourier_pts = cp.ascontiguousarray(
            cp.mod(cp.asarray(fourier_pts, dtype=self.dtype) + cp.pi, 2 * cp.pi) - cp.pi
        )

        self.num_pts = self.fourier_pts.shape[1]
        self.epsilon = max(epsilon, np.finfo(self.dtype).eps)

        self._transform_plan = cufPlan(
            2, self.sz, self.ntransforms, self.epsilon, -1, dtype=self.complex_dtype
        )

        self.adjoint_opts = dict()
        if self.dtype is np.float64 and self.dim == 3 and self.epsilon < 1e3:
            # Note this is an algorithmic implementation dictated by shmem.
            logger.info(
                "Converting cufinufft gpu_method=1 from default of 2 for 3D1 transform,"
                f"to support computation in double precision with tol={self.epsilon}."
            )
            self.adjoint_opts["gpu_method"] = 1

        self._adjoint_plan = cufPlan(
            1,
            self.sz,
            self.ntransforms,
            self.epsilon,
            1,
            dtype=self.complex_dtype,
            **self.adjoint_opts,
        )

        self._transform_plan.setpts(*self.fourier_pts)
        self._adjoint_plan.setpts(*self.fourier_pts)

    def transform(self, signal):
        """
        Compute the NUFFT transform using this plan instance.

        :param signal: Signal to be transformed. For a single transform,
        this should be a a 1, 2, or 3D array matching the plan `sz`.
        For a batch, signal should have shape `(*sz, ntransforms)`.

        :returns: Transformed signal of shape `num_pts` or
        `(ntransforms, num_pts)` as CuPy array.
        """

        # Check we're not forcing a dtype workaround for ASPIRE-Python/703,
        #   then check if we have a dtype mismatch.
        # This avoids false positive complaint for the workaround.
        if (self._original_dtype == self.dtype) and not (
            signal.dtype == self.dtype or signal.dtype == self.complex_dtype
        ):
            logger.warning(
                "Incorrect dtypes passed to (a)nufft."
                " In the future this will be an error."
            )

        # Note, if not C order, cuFINUFFT will copy-cast anyway.
        signal = cp.asarray(signal, order="C", dtype=self.complex_dtype)

        sig_shape = signal.shape
        res_shape = self.num_pts
        # Note, there is a corner case for ntransforms == 1.
        if self.ntransforms > 1 or (
            self.ntransforms == 1 and len(signal.shape) == self.dim + 1
        ):
            assert (
                len(signal.shape) == self.dim + 1
            ), f"For multiple transforms, {self.dim}D signal should be a {self.ntransforms} element stack of {self.sz}."

            assert (
                signal.shape[0] == self.ntransforms
            ), "For multiple transforms, signal stack length should match ntransforms {self.ntransforms}."

            sig_shape = signal.shape[1:]  # order...
            res_shape = (self.ntransforms, self.num_pts)

        assert (
            sig_shape == self.sz
        ), f"Signal frame to be transformed must have shape {self.sz}"

        result = cp.empty(res_shape, dtype=self.complex_dtype)

        if signal.dtype != self.complex_dtype:
            signal = signal.astype(self.complex_dtype)

        self._transform_plan.execute(signal, out=result)

        # ASPIRE-Python/703
        if result.dtype != complex_type(self._original_dtype):
            result = result.astype(complex_type(self._original_dtype))

        return result

    def adjoint(self, signal):
        """
        Compute the NUFFT adjoint using this plan instance.

        :param signal: Signal to be transformed. For a single transform,
        this should be a a 1D array of len `num_pts`.
        For a batch, signal should have shape `(ntransforms, num_pts)`.

        :returns: Transformed signal `(sz)` or `(sz, ntransforms)` as CuPy array.
        """

        # Check we're not forcing a dtype workaround for ASPIRE-Python/703,
        #   then check if we have a dtype mismatch.
        # This avoids false positive complaint for the workaround.
        if (self._original_dtype == self.dtype) and not (
            signal.dtype == self.complex_dtype or signal.dtype == self.dtype
        ):
            logger.warning(
                "Incorrect dtypes passed to (a)nufft."
                " In the future this will be an error."
            )

        # Note, if not C order, cuFINUFFT will copy-cast anyway.
        signal = cp.asarray(signal, order="C", dtype=self.complex_dtype)

        res_shape = self.sz
        # Note, there is a corner case for ntransforms == 1.
        if self.ntransforms > 1 or (self.ntransforms == 1 and len(signal.shape) == 2):
            assert (
                len(signal.shape) == 2
            ), f"For multiple {self.dim}D adjoints, signal should be a {self.ntransforms} element stack of {self.num_pts}."
            assert (
                signal.shape[0] == self.ntransforms
            ), "For multiple transforms, signal stack length should match ntransforms {self.ntransforms}."
            res_shape = (self.ntransforms, *self.sz)

        result = cp.empty(res_shape, dtype=self.complex_dtype)

        if signal.dtype != self.complex_dtype:
            signal = signal.astype(self.complex_dtype)

        self._adjoint_plan.execute(signal, out=result)

        # ASPIRE-Python/703
        if result.dtype != complex_type(self._original_dtype):
            result = result.astype(complex_type(self._original_dtype))

        return result
