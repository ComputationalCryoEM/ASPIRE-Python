import numpy as np
from pynfft.nfft import NFFT

from aspire.nufft import Plan
from aspire.nufft.utils import nextpow2
from aspire.utils import ensure


class PyNfftPlan(Plan):
    @staticmethod
    def epsilon_to_nfft_cutoff(epsilon):
        # NOTE: These are obtained empirically. Should have a theoretical derivation.
        rel_errs = [6e-2, 2e-3, 2e-5, 2e-7, 3e-9, 4e-11, 4e-13, 0]
        return list(
            filter(lambda i_err: i_err[1] < epsilon, enumerate(rel_errs, start=1))
        )[0][0]

    def __init__(self, sz, fourier_pts, epsilon=1e-15, **kwargs):
        """
        A plan for non-uniform FFT (3D)
        :param sz: A tuple indicating the geometry of the signal
        :param fourier_pts: The points in Fourier space where the Fourier transform is to be calculated,
            arranged as a 3-by-K array. These need to be in the range [-pi, pi] in each dimension.
        :param epsilon: The desired precision of the NUFFT
        """
        self.sz = sz
        self.dim = len(sz)
        self.fourier_pts = fourier_pts
        self.num_pts = fourier_pts.shape[1]
        self.epsilon = epsilon

        self.cutoff = PyNfftPlan.epsilon_to_nfft_cutoff(epsilon)
        self.multi_bandwith = tuple(2 * 2 ** nextpow2(self.sz))
        # TODO - no other flags used in the MATLAB code other than these 2 are supported by the PyNFFT wrapper
        self._flags = ("PRE_PHI_HUT", "PRE_PSI")

        self._plan = NFFT(
            N=self.sz,
            M=self.num_pts,
            n=self.multi_bandwith,
            m=self.cutoff,
            flags=self._flags,
        )

        self._plan.x = ((1.0 / (2 * np.pi)) * self.fourier_pts).T
        self._plan.precompute()

    def transform(self, signal):
        ensure(
            signal.shape == self.sz,
            f"Signal to be transformed must have shape {self.sz}",
        )

        self._plan.f_hat = signal.astype("complex64")
        f = self._plan.trafo()

        if signal.dtype == np.float32:
            f = f.astype("complex64")

        return f

    def adjoint(self, signal):
        self._plan.f = signal.astype("complex64")
        f_hat = self._plan.adjoint()

        if signal.dtype == np.float32:
            f_hat = f_hat.astype("complex64")

        return f_hat
