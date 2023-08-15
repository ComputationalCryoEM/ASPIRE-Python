import logging

import numpy as np

from aspire.numeric import fft
from aspire.utils.matlab_compat import m_reshape
from aspire.volume import Volume

logger = logging.getLogger(__name__)


class Kernel:
    pass


class FourierKernel(Kernel):
    def __init__(self, kernel):
        self.ndim = kernel.ndim
        self.kernel = kernel
        self.M = kernel.shape[0]
        self.dtype = kernel.dtype

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
        return FourierKernel(new_kernel)

    def circularize(self):
        logger.info("Circularizing kernel")
        kernel = np.real(fft.ifftn(self.kernel))
        kernel = fft.mdim_fftshift(kernel)

        for dim in range(self.ndim):
            logger.info(f"Circularizing dimension {dim}")
            kernel = self.circularize_1d(kernel, dim)

        xx = fft.fftn(fft.mdim_ifftshift(kernel)).real
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

        return fft.fftshift(kernel_circ, dim)

    def convolve_volume(self, x, in_place=False):
        """
        Convolve volume with kernel

        :param x: A Volume instance
        :param in_plane: Operate on Volume `x` in place.  Optional bool, defaults False.
            This saves memory in exchange for mutating the input data.
        :return: Volume instance convolved by the kernel with the same dimensions as before.
        """

        kernel_f = self.kernel[..., np.newaxis]
        return self._convolve_volume(x, kernel_f, in_place=in_place)

    def _convolve_volume(self, x, kernel_f, in_place=False):
        """
        Private method for convolving volume with kernel_f.

        :param x: A Volume instance
        :param kernel_f: Kernel as numpy array.
        :param in_plane: Operate on Volume `x` in place.  Optional bool, defaults False.
            This saves memory in exchange for mutating the input data.
        :return: Volume instance convolved by the kernel with the same dimensions as before.
        """

        if not isinstance(x, Volume):
            x = Volume(x)

        N = x.resolution
        N_ker = kernel_f.shape[0]

        assert kernel_f.shape[3] == 1, "Convolution kernel must be cubic"
        assert len(set(kernel_f.shape[:3])) == 1, "Convolution kernel must be cubic"

        is_singleton = len(x) == 1

        if is_singleton:
            pad_width = [(0, N_ker - N)] * 3
            _x = np.pad(x.asnumpy()[0], pad_width)
            x_f = fft.fftn(_x)[..., np.newaxis]
        else:
            raise NotImplementedError("not yet")

        x_f = x_f * kernel_f

        # `in_place` mutates the original volume
        if not in_place:
            x = Volume.empty_like(x)

        if is_singleton:
            x[0] = np.real(fft.ifftn(x_f[..., 0])[:N, :N, :N])
        else:
            raise NotImplementedError("not yet")

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
        assert (
            len(set(shape[i] for i in range(5))) == 1
        ), "Volume matrix must be cubic and square"

        # TODO from MATLAB code: Deal with rolled dimensions
        N_ker = kernel_f.shape[0]

        # Note from MATLAB code:
        # Order is important here.  It's about 20% faster to run from 1 through 6 compared with 6 through 1.
        # TODO: Experiment with fft axis order
        for i in range(6):
            _pad_width = [(0, 0)] * 6
            _pad_width[i] = (0, N_ker - N)
            x = fft.fft(np.pad(x, _pad_width), axis=i)

        x *= kernel_f

        indices = list(range(N))
        for i in range(5, -1, -1):
            x = fft.ifft(x, axis=i)
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

        A = Volume(np.eye(L**3, dtype=self.dtype).reshape((L**3, L, L, L)))
        for i in range(L**3):
            A[i] = self.convolve_volume(A[i])[0]

        A = A.asnumpy().reshape((L,) * 6)

        return A


class FourierKernelMatrix(FourierKernel):
    def __init__(self, kermat):
        self.ndim = kermat.ndim - 2
        self.kermat = kermat
        self.r = kermat.shape[0]
        assert kermat.shape[1] == self.r
        self.dtype = kermat.dtype
        self.M = kermat.shape[-1]

    def __add__(self, delta):
        new_kermat = self.kermat + delta
        return FourierKernelMatrix(new_kermat)

    def circularize(self):
        _L = self.M // 2
        xx = np.empty((self.r, self.r, _L, _L, _L))
        for k in range(self.r):
            for j in range(self.r):
                xx[k, j] = FourierKernel(self.kermat[k, j]).circularize().real
        return xx

    def convolve_volume(self, x, k, j, in_place=False):
        """
        Convolve volume with kernel

        :param x: A Volume instance
        :param k: Kernel matrix index
        :param j: Kernel matrix index
        :param in_plane: Operate on Volume `x` in place.  Optional bool, defaults False.
            This saves memory in exchange for mutating the input data.
        :return: Volume instance convolved by the kernel with the same dimensions as before.
        """

        kernel_f = self.kermat[k, j, ..., np.newaxis]
        return self._convolve_volume(x, kernel_f, in_place=in_place)

    def convolve_volume_matrix(self, x):
        raise NotImplementedError("Not implemented for Fourier Kernel Matrix")

    def toeplitz(self, L=None):
        if L is None:
            L = int(self.M / 2)

        Amat = np.empty((self.r, self.r, self.L, self.L, self.L))
        for k in range(self.r):
            for j in range(self.r):
                Amat[k, j] = FourierKernel(self.kermat[k, j]).toeplitz(L)

        return Amat
