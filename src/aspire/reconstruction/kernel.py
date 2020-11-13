import logging

import numpy as np
from scipy.fftpack import fft, fftn, fftshift, ifft, ifftn

from aspire.utils import (
    ensure,
    roll_dim,
    unroll_dim,
    vec_to_vol,
    vecmat_to_volmat,
    vol_to_vec,
)
from aspire.utils.fft import mdim_fftshift, mdim_ifftshift
from aspire.utils.matlab_compat import m_reshape

logger = logging.getLogger(__name__)


class Kernel:
    pass


class FourierKernel(Kernel):
    def __init__(self, kernel, centered):
        self.ndim = kernel.ndim
        self.kernel = kernel
        self.M = kernel.shape[0]
        self.dtype = kernel.dtype

        # TODO: `centered` should be populated based on how the object is constructed, not explicitly
        self._centered = centered

    def __add__(self, delta):
        """
        Add a tiny delta to the underlying kernel.

        :param delta: A scalar or an `ndarray` that can be broadcast to the `kernel` attribute of this object.
        :return: A new FourierKernel object with a modified kernel

        .. note::
            There is often a need to add a regularization parameter (a small positive value) to a FourierKernel object,
            to be able to use it within optimization loops. This operator allows one to use the FourierKernel object
            with the underlying 'kernel' attribute tweaked with a regularization parameter.
        """
        new_kernel = self.kernel + delta
        return FourierKernel(new_kernel, self._centered)

    def is_centered(self):
        return self._centered

    def circularize(self):
        logger.info("Circularizing kernel")
        kernel = np.real(ifftn(self.kernel))
        kernel = mdim_fftshift(kernel)

        for dim in range(self.ndim):
            logger.info(f"Circularizing dimension {dim}")
            kernel = self.circularize_1d(kernel, dim)

        xx = fftn(mdim_ifftshift(kernel))
        return xx

    def circularize_1d(self, kernel, dim):
        ndim = kernel.ndim
        sz = kernel.shape
        N = int(sz[dim] / 2)

        top, bottom = np.split(kernel, 2, axis=dim)

        # Multiplier for weighted average
        mult_shape = [1] * ndim
        mult_shape[dim] = N
        mult_shape = tuple(mult_shape)

        mult = m_reshape((np.arange(N, dtype=self.dtype) / N), mult_shape)
        kernel_circ = mult * top

        mult = m_reshape((np.arange(N, 0, -1, dtype=self.dtype) / N), mult_shape)
        kernel_circ += mult * bottom

        return fftshift(kernel_circ, dim)

    def convolve_volume(self, x):
        """
        Convolve volume with kernel
        :param x: An N-by-N-by-N-by-... array of volumes to be convolved.
        :return: The original volumes convolved by the kernel with the same dimensions as before.
        """
        N = x.shape[0]
        kernel_f = self.kernel[..., np.newaxis]
        N_ker = kernel_f.shape[0]

        x, sz_roll = unroll_dim(x, 4)
        ensure(
            x.shape[0] == x.shape[1] == x.shape[2] == N, "Volumes in x must be cubic"
        )
        ensure(kernel_f.shape[3] == 1, "Convolution kernel must be cubic")
        ensure(len(set(kernel_f.shape[:3])) == 1, "Convolution kernel must be cubic")

        is_singleton = x.shape[3] == 1

        if is_singleton:
            x = fftn(x[..., 0], (N_ker, N_ker, N_ker))[..., np.newaxis]
        else:
            raise NotImplementedError("not yet")

        x = x * kernel_f

        if is_singleton:
            x[..., 0] = np.real(ifftn(x[..., 0]))
            x = x[:N, :N, :N, :]
        else:
            raise NotImplementedError("not yet")

        x = roll_dim(x, sz_roll)

        return x

    def convolve_volume_matrix(self, x):
        """
        Convolve volume matrix with kernel
        :param x: An N-by-...-by-N (6 dimensions) volume matrix to be convolved.
        :return: The original volume matrix convolved by the kernel with the same dimensions as before.
        """
        shape = x.shape
        N = shape[0]
        kernel_f = self.kernel
        ensure(
            len(set(shape[i] for i in range(5))) == 1,
            "Volume matrix must be cubic and square",
        )

        # TODO from MATLAB code: Deal with rolled dimensions
        N_ker = kernel_f.shape[0]

        # Note from MATLAB code:
        # Order is important here.  It's about 20% faster to run from 1 through 6 compared with 6 through 1.
        # TODO: Experiment with scipy order; try overwrite_x argument
        for i in range(6):
            x = fft(x, N_ker, i, overwrite_x=True)

        x *= kernel_f

        indices = list(range(N))
        for i in range(5, -1, -1):
            x = ifft(x, None, i, overwrite_x=True)
            x = x.take(indices, axis=i)

        return np.real(x)

    def toeplitz(self, L=None):
        """
        Compute the 3D Toeplitz matrix corresponding to this Fourier Kernel
        :param L: The size of the volumes to be convolved (default M/2, where the dimensions of this Fourier Kernel
            are MxMxM
        :return: An six-dimensional Toeplitz matrix of size L describing the convolution of a volume with this kernel
        """
        if L is None:
            L = int(self.M / 2)

        A = np.eye(L ** 3, dtype=self.dtype)
        for i in range(L ** 3):
            A[:, i] = np.real(vol_to_vec(self.convolve_volume(vec_to_vol(A[:, i]))))

        A = vecmat_to_volmat(A)
        return A
