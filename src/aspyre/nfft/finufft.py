import numpy as np
import finufftpy
from aspyre.nfft import Plan
from aspyre.utils import ensure


class FINufftPlan(Plan):

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
        # TODO: Things get messed up unless we ensure a 'C' ordering here - investigate why
        self.fourier_pts = np.asarray(np.mod(fourier_pts + np.pi, 2 * np.pi) - np.pi, order='C')
        self.num_pts = fourier_pts.shape[1]
        self.epsilon = epsilon

        # Get a handle on the appropriate 1d/2d/3d forward transform function in finufftpy
        self.transform_function = getattr(finufftpy, {1: 'nufft1d2', 2: 'nufft2d2', 3: 'nufft3d2'}[self.dim])
        # Get a handle on the appropriate 1d/2d/3d adjoint function in finufftpy
        self.adjoint_function = getattr(finufftpy, {1: 'nufft1d1', 2: 'nufft2d1', 3: 'nufft3d1'}[self.dim])

    def transform(self, signal):
        ensure(signal.shape == self.sz, f'Signal to be transformed must have shape {self.sz}')

        epsilon = max(self.epsilon, np.finfo(signal.dtype).eps)

        # Forward transform functions in finufftpy have signatures of the form:
        # (x, y, z, c, isign, eps, f, ...)
        # (x, y     c, isign, eps, f, ...)
        # (x,       c, isign, eps, f, ...)
        # Where f is a Fortran-order ndarray of the appropriate dimensions
        # We form these function signatures here by tuple-unpacking

        result = np.zeros(self.num_pts).astype('complex128')

        result_code = self.transform_function(
            *self.fourier_pts,
            result,
            -1,
            epsilon,
            signal
        )

        if result_code != 0:
            raise RuntimeError(f'FINufft transform failed. Result code {result_code}')

        return result

    def adjoint(self, signal):

        epsilon = max(self.epsilon, np.finfo(signal.dtype).eps)

        # Adjoint functions in finufftpy have signatures of the form:
        # (x, y, z, c, isign, eps, ms, mt, mu, f, ...)
        # (x, y     c, isign, eps, ms, mt      f, ...)
        # (x,       c, isign, eps, ms,         f, ...)
        # Where f is a Fortran-order ndarray of the appropriate dimensions
        # We form these function signatures here by tuple-unpacking

        # Note: Important to have order='F' here!
        result = np.zeros(self.sz, order='F').astype('complex128')

        result_code = self.adjoint_function(
            *self.fourier_pts,
            signal,
            1,
            epsilon,
            *self.sz,
            result
        )
        if result_code != 0:
            raise RuntimeError(f'FINufft adjoint failed. Result code {result_code}')

        return result
