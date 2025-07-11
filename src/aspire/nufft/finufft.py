import logging

import finufft
import numpy as np

from aspire.nufft import Plan
from aspire.utils import complex_type

logger = logging.getLogger(__name__)


class FinufftPlan(Plan):
    def __init__(self, sz, fourier_pts, epsilon=1e-8, ntransforms=1, **kwargs):
        """
        A plan for non-uniform FFT in 2D or 3D.

        :param sz: A tuple indicating the geometry of the signal.
        :param fourier_pts: The points in Fourier space where the Fourier
            transform is to be calculated, arranged as a dimension-by-K array.
            These need to be in the range [-pi, pi] in each dimension.
        :param epsilon: The desired precision of the NUFFT.
        :param ntransforms: Optional integer indicating if you would like
            to compute a batch of `ntransforms`.
            transforms.  Implies vol_f.shape is (`ntransforms`, ...).
        """

        self.ntransforms = ntransforms

        self.sz = sz
        self.dim = len(sz)

        self.dtype = fourier_pts.dtype

        self.complex_dtype = complex_type(self.dtype)

        self.fourier_pts = np.ascontiguousarray(
            np.mod(fourier_pts + np.pi, 2 * np.pi) - np.pi
        )

        self.num_pts = fourier_pts.shape[1]

        self.epsilon = max(epsilon, np.finfo(self.dtype).eps)
        if self.epsilon != epsilon:
            logger.debug(
                f"FinufftPlan adjusted eps={self.epsilon}" f" from requested {epsilon}."
            )

        self._transform_plan = finufft.Plan(
            nufft_type=2,
            n_modes_or_dim=self.sz,
            eps=self.epsilon,
            n_trans=self.ntransforms,
            dtype=self.complex_dtype,
            upsampfac=2,  # revert <2.4.0 default
        )

        self._adjoint_plan = finufft.Plan(
            nufft_type=1,
            n_modes_or_dim=self.sz,
            eps=self.epsilon,
            n_trans=self.ntransforms,
            dtype=self.complex_dtype,
            upsampfac=2,  # revert <2.4.0 default
        )

        self._transform_plan.setpts(*self.fourier_pts)
        self._adjoint_plan.setpts(*self.fourier_pts)

    def transform(self, signal):
        """
        Compute the NUFFT transform using this plan instance.

        :param signal: Signal to be transformed. For a single transform,
            this should be a a 1, 2, or 3D array matching the plan `sz`.
            For a batch, signal should have shape `(ntransforms, *sz)`.

        :returns: Transformed signal of shape `num_pts` or
            `(ntransforms, num_pts)`.
        """

        sig_frame_shape = signal.shape
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

            sig_frame_shape = signal.shape[1:]

            # finufft expects signal.ndim == dim for ntransforms = 1.
            if self.ntransforms == 1:
                signal = signal.reshape(self.sz)

        assert (
            sig_frame_shape == self.sz
        ), f"Signal frame to be transformed must have shape {self.sz}"

        # FINUFFT was designed for a complex input array
        signal = np.asarray(signal, dtype=self.complex_dtype, order="C")

        result = self._transform_plan.execute(signal)

        return result

    def adjoint(self, signal):
        """
        Compute the NUFFT adjoint using this plan instance.

        :param signal: Signal to be transformed. For a single transform,
            this should be a a 1D array of len `num_pts`.
            For a batch, signal should have shape `(ntransforms, num_pts)`.

        :returns: Transformed signal `(sz)` or `(ntransforms, sz)`.
        """

        # Note, there is a corner case for ntransforms == 1.
        if self.ntransforms > 1 or (self.ntransforms == 1 and len(signal.shape) == 2):
            assert (
                len(signal.shape) == 2
            ), f"For multiple {self.dim}D adjoints, signal should be a {self.ntransforms} element stack of {self.num_pts}."
            assert (
                signal.shape[0] == self.ntransforms
            ), "For multiple transforms, signal stack length should match ntransforms {self.ntransforms}."

            # finufft is expecting flat array for 1D case.
            if self.ntransforms == 1:
                signal = signal.reshape(self.num_pts)

        # FINUFFT was designed for a complex input array
        signal = np.asarray(signal, dtype=self.complex_dtype, order="C")

        result = self._adjoint_plan.execute(signal)

        return result
