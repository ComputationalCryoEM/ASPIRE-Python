import logging

import finufftpy
import numpy as np

from aspire.nufft import Plan
from aspire.utils import complex_type, ensure

logger = logging.getLogger(__name__)


class FinufftPlan(Plan):
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

        self.ntransforms = ntransforms
        manystr = ""
        if self.ntransforms > 1:
            manystr = "many"

        self.sz = sz
        self.dim = len(sz)

        self.dtype = fourier_pts.dtype

        # TODO: Currently/historically finufftpy is hardcoded as doubles only (inside the binding layer).
        #       This has been changed in their GuruV2 work,
        #         and an updated package should be released and integrated very soon.
        #       The following casting business is only to facilitate the transition.
        #       I don't want to send one precision in and get a different one out.
        #         We have enough code that does that already.
        #         (Potentially anything that used this, for example).
        #       I would error, but I know a bunch of code ASPIRE wants to work,
        #         the cov2d tutorial for example, would fail and require hacks to run
        #         if I was strict about dtypes rn.
        #       I would rather deal with that in other, more targeted PRs.
        #       This preserves the legacy behavior of admitting singles,
        #         but I will correct it slightly, to return the precision given as input.
        #       This approach should contain the hacks here to a single place on the edge,
        #         instead of spread through the code. ASPIRE code should focus on being
        #         internally consistent.
        #       Admittedly not ideal, but ignoring these problems wasn't sustainable.

        self.cast_output = False
        if self.dtype != np.float64:
            logger.debug(
                "This version of finufftpy is hardcoded to doubles internally"
                "  casting input to doubles, results cast back to singles."
            )
            self.cast_output = True
            self.dtype = np.float64

        self.complex_dtype = complex_type(self.dtype)

        # TODO: Things get messed up unless we ensure a 'C' ordering here - investigate why
        self.fourier_pts = np.asarray(
            np.mod(fourier_pts + np.pi, 2 * np.pi) - np.pi, order="C", dtype=self.dtype
        )
        self.num_pts = fourier_pts.shape[1]
        self.epsilon = epsilon

        # Get a handle on the appropriate 1d/2d/3d forward transform function in finufftpy
        self.transform_function = getattr(finufftpy, f"nufft{self.dim}d2{manystr}")

        # Get a handle on the appropriate 1d/2d/3d adjoint function in finufftpy
        self.adjoint_function = getattr(finufftpy, f"nufft{self.dim}d1{manystr}")

    def transform(self, signal):
        """
        Compute the NUFFT transform using this plan instance.

        :param signal: Signal to be transformed. For a single transform,
        this should be a a 1, 2, or 3D array matching the plan `sz`.
        For a batch, signal should have shape `(*sz, ntransforms)`.

        :returns: Transformed signal of shape `num_pts` or
        `(ntransforms, num_pts)`.
        """

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

            sig_shape = signal.shape[1:]
            res_shape = (self.ntransforms, self.num_pts)

        ensure(
            sig_shape == self.sz,
            f"Signal frame to be transformed must have shape {self.sz}",
        )

        epsilon = max(self.epsilon, np.finfo(signal.dtype).eps)

        # Forward transform functions in finufftpy have signatures of the form:
        # (x, y, z, c, isign, eps, f, ...)
        # (x, y     c, isign, eps, f, ...)
        # (x,       c, isign, eps, f, ...)
        # Where f is a Fortran-order ndarray of the appropriate dimensions
        # We form these function signatures here by tuple-unpacking

        result = np.zeros(res_shape, dtype=self.complex_dtype)

        result_code = self.transform_function(
            *self.fourier_pts,
            result,
            -1,
            epsilon,
            signal.T,  # RCOPT, currently F ordered, should change in gv2
        )

        if result_code != 0:
            raise RuntimeError(f"FINufft transform failed. Result code {result_code}")
        if self.cast_output:
            result = result.astype(np.complex64)

        return result

    def adjoint(self, signal):
        """
        Compute the NUFFT adjoint using this plan instance.

        :param signal: Signal to be transformed. For a single transform,
        this should be a a 1D array of len `num_pts`.
        For a batch, signal should have shape `(ntransforms, num_pts)`.

        :returns: Transformed signal `(sz)` or `(sz, ntransforms)`.
        """

        epsilon = max(self.epsilon, np.finfo(signal.dtype).eps)

        # Adjoint functions in finufftpy have signatures of the form:
        # (x, y, z, c, isign, eps, ms, mt, mu, f, ...)
        # (x, y     c, isign, eps, ms, mt      f, ...)
        # (x,       c, isign, eps, ms,         f, ...)
        # Where f is a Fortran-order ndarray of the appropriate dimensions
        # We form these function signatures here by tuple-unpacking

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
            res_shape = (
                self.ntransforms,
                *self.sz,
            )

        result = np.zeros(res_shape, dtype=self.complex_dtype)

        # FINUFFT is F order at this time. The bindings
        # will pickup the fact `signal` is C_Contiguous,
        # and transpose the data; we just need to transpose
        # the indices.  I think the next release addresses this.
        #   Note in the 2020 hackathon this was changed directly in FFB,
        #   which worked because GPU arrays just need the pointer anyway...
        #   This is a quirk of this version of FINUFFT, and
        #   so probably belongs here at the edge,
        #   away from other implementations.
        signal = signal.reshape(signal.shape[::-1])

        result_code = self.adjoint_function(
            *self.fourier_pts, signal, 1, epsilon, *self.sz, result
        )
        if result_code != 0:
            raise RuntimeError(f"FINufft adjoint failed. Result code {result_code}")

        if self.cast_output:
            result = result.astype(np.complex64)

        return result
