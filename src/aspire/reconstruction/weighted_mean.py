import logging
from functools import partial

import numpy as np
import scipy.sparse.linalg
from scipy.fftpack import fft2
from scipy.linalg import norm
from scipy.sparse.linalg import LinearOperator

from aspire import config
from aspire.image import Image
from aspire.nufft import anufft
from aspire.reconstruction import FourierKernel, FourierKernelMat, MeanEstimator
from aspire.utils import vec_to_symmat_iso, vecmat_to_volmat, volmat_to_vecmat
from aspire.utils.fft import mdim_ifftshift
from aspire.utils.matlab_compat import m_flatten, m_reshape
from aspire.volume import Volume, rotated_grids

logger = logging.getLogger(__name__)


class WeightedVolumesEstimator(MeanEstimator):
    def __init__(self, weights, *args, **kwargs):
        """
        Weighted mean volume estimation.

        This is best considered as an r x r matrix of volumes;
        each volume is a weighted mean least-squares estimator kernel (MeanEstimator).
        Convolution with each of thesekernels is equivalent
        to performing a projection/backprojection on a volume,
        with the appropriate amplitude modifiers and CTF,
        and also a weighting term;
        the r^2 volumes are each of pairwise products between the weighting vectors given by the columns of wts.

        Note that this is a non-centered Fourier transform, so the zero frequency is found at index 1.

        :param weights: Matrix of weights, n x r.
        """

        self.weights = weights
        self.r = self.weights.shape[1]
        super().__init__(*args, **kwargs)
        assert self.n == self.weights.shape[0]

    def __getattr__(self, name):
        if name == "precond_kernel":
            if self.preconditioner == "circulant":
                precond_kernel = 1.0 / self.kernel.circularize()
                print(precond_kernel)
                print("precond_kernel.shape", precond_kernel.shape)
                precond_kernel = self.precond_kernel = FourierKernelMat(precond_kernel, centered=True)
            else:
                precond_kernel = self.precond_kernel = None
            return precond_kernel
        
        return super().__getattr__(name)

    def compute_kernel(self):
        """
        :return: r x r matrix of volumes, shaped (r, r, 2L, 2L, 2L).
        """

        _2L = 2 * self.L
        kernel = np.zeros((self.r, self.r, _2L, _2L, _2L), dtype=self.dtype)
        sq_filters_f = self.src.eval_filter_grid(self.L, power=2)

        for k in range(self.r):
            for j in range(k):
                for i in range(0, self.n, self.batch_size):
                    _range = np.arange(
                        i, min(self.n, i + self.batch_size), dtype=np.int
                    )
                    pts_rot = rotated_grids(self.L, self.src.rots[_range, :, :])
                    weights = sq_filters_f[:, :, _range]
                    weights *= self.src.amplitudes[_range] ** 2

                    if self.L % 2 == 0:
                        weights[0, :, :] = 0
                        weights[:, 0, :] = 0

                    pts_rot = m_reshape(pts_rot, (3, -1))
                    weights = m_flatten(weights)

                    kernel[k, j] += (
                        1
                        / (self.n * self.L ** 4)
                        * anufft(weights, pts_rot, (_2L, _2L, _2L), real=True)
                    )
                    # r x r symmetric
                    kernel[j, k] = kernel[j, k]

        # Ensure symmetric kernel
        kernel[:, :, 0, :, :] = 0
        kernel[:, :, :, 0, :] = 0
        kernel[:, :, :, :, 0] = 0

        kermat_f = np.empty((self.r, self.r, _2L, _2L, _2L))
        logger.info("Computing non-centered Fourier Transform Kernel Mat")
        for k in range(self.r):
            for j in range(k):
                kernel[k, j] = mdim_ifftshift(kernel[k, j], range(0, 3))
                # should this be fft3?
                print("here", kernel[k, j].shape)
                kernel_f = fft2(kernel[k, j], axes=(0, 1, 2))
                kernel_f = np.real(kernel_f)
                print("XXX", kernel_f.shape)
                kermat_f[k, j] = kernel_f
                kermat_f[j, k] = kermat_f[k, j]

        return FourierKernelMat(kermat_f, centered=False)

    def src_backward(self):
        # src_vols_wt_backward

        mean_b = np.zeros((self.r, self.L, self.L, self.L), dtype=self.dtype)

        for k in range(self.r):
            for i in range(0, self.n, self.batch_size):
                im = self.src.images(i, self.batch_size)

                batch_mean_b = self.src.im_backward(im, i, self.weights[:, k]) / self.n
                mean_b[k] += batch_mean_b.astype(self.dtype)

        res = np.sqrt(self.n) * self.basis.evaluate_t(mean_b)
        logger.info(f"Determined weighted adjoint mappings. Shape = {res.shape}")

        return res

    def conj_grad(self, b_coeff, tol=None):
        print("cg", b_coeff.shape)
        n = b_coeff.shape[-1]
        kernel = self.kernel

        regularizer = config.mean.regularizer
        if regularizer > 0:
            kernel += regularizer

        operator = LinearOperator(
            (self.r * n, self.r * n),
            matvec=partial(self.apply_kernel, kernel=kernel),
            dtype=self.dtype,
        )
        if self.precond_kernel is None:
            M = None
        else:
            precond_kernel = self.precond_kernel
            if regularizer > 0:
                precond_kernel += regularizer
            M = LinearOperator(
                (self.r * n, self.r * n),
                matvec=partial(self.apply_kernel, kernel=precond_kernel),
                dtype=self.dtype,
            )

        tol = tol or config.mean.cg_tol
        target_residual = tol * norm(b_coeff)

        def cb(xk):
            logger.info(
                f"Delta {norm(b_coeff - self.apply_kernel(xk))} (target {target_residual})"
            )

        print("b_coeff.shape", b_coeff.shape)
        x, info = scipy.sparse.linalg.cg(
            operator, b_coeff.flatten(), M=M, callback=cb, tol=tol, atol=0
        )

        if info != 0:
            raise RuntimeError("Unable to converge!")

        # Thinking might be clearer if r x ... but would need to mess with roll/unroll in FBB.
        # return x.reshape(self.r, -1)
        return x


    def apply_kernel(self, vol_coeff, kernel=None):
        """
        Applies the kernel represented by convolution
        :param vol_coeff: The volume to be convolved, stored in the basis coefficients.
        :param kernel: a Kernel object. If None, the kernel for this Estimator is used.
        :return: The result of evaluating `vol_coeff` in the given basis, convolving with the kernel given by
            kernel, and backprojecting into the basis.
        """
        if kernel is None:
            kernel = self.kernel

        assert np.size(vol_coeff) == self.r * self.basis.count
        if vol_coeff.ndim == 1:
            vol_coeff = vol_coeff.reshape(self.r, self.basis.count)

        vols_out = np.zeros((self.r, self.L, self.L, self.L), dtype=self.dtype)
        print(f"vol_coeff.shape {vol_coeff.shape}")  # 2 98

        for k in range(self.r):
            vol = self.basis.evaluate(vol_coeff[k])
            print("vol.shape", vol.shape)  # 8 8 8
            for j in range(self.r):
                print("convolv", kernel.convolve_volume(vol, k, j).shape)  # (8, 8, 8)
                vols_out[k] += kernel.convolve_volume(vol, k, j)

        vol_coeff = self.basis.evaluate_t(vols_out)

        return vol_coeff
