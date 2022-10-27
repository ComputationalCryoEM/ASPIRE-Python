import logging
from functools import partial

import numpy as np
import scipy.sparse.linalg
from scipy.linalg import norm
from scipy.sparse.linalg import LinearOperator

from aspire import config
from aspire.reconstruction.kernel import FourierKernel
from aspire.volume import Volume

logger = logging.getLogger(__name__)


class Estimator:
    def __init__(self, src, basis, batch_size=512, preconditioner="circulant"):
        """
        An object representing a 2*L-by-2*L-by-2*L array containing the non-centered Fourier transform of the mean
        least-squares estimator kernel.
        Convolving a volume with this kernel is equal to projecting and backproject-ing that volume in each of the
        projection directions (with the appropriate amplitude multipliers and CTFs) and averaging over the whole
        dataset.
        Note that this is a non-centered Fourier transform, so the zero frequency is found at index 1.
        """

        self.src = src
        self.basis = basis
        self.dtype = self.src.dtype
        self.batch_size = batch_size
        self.preconditioner = preconditioner

        if not self.dtype == self.basis.dtype:
            logger.warning(
                f"Inconsistent types in {self.dtype} Estimator."
                f" basis: {self.basis.dtype}"
            )

        if src.L != basis.nres:
            raise ValueError(
                "Currently require 2D source and 3D volume resolution to be the same."
                f" Given src.L={src.L} != {basis.nres}"
            )

    def __getattr__(self, name):
        """Lazy attributes instantiated on first-access"""

        if name == "kernel":
            logger.info("Computing kernel")
            kernel = self.kernel = self.compute_kernel()
            return kernel

        elif name == "precond_kernel":
            if self.preconditioner == "circulant":
                logger.info("Computing Preconditioner kernel")
                precond_kernel = self.precond_kernel = FourierKernel(
                    1.0 / self.kernel.circularize(), centered=True
                )
            else:
                precond_kernel = self.precond_kernel = None
            return precond_kernel

        return super(Estimator, self).__getattr__(name)

    def compute_kernel(self):
        raise NotImplementedError("Subclasses must implement the compute_kernel method")

    def estimate(self, b_coeff=None, tol=None):
        """Return an estimate as a Volume instance."""
        if b_coeff is None:
            b_coeff = self.src_backward()
        # conj_grad expects a 1d array if n = 1
        b_coeff = np.squeeze(b_coeff, axis=0)
        est_coeff = self.conj_grad(b_coeff, tol=tol)
        est = self.basis.evaluate(est_coeff).T

        return est

    def src_backward(self):
        """
        Apply adjoint mapping to source

        :return: The adjoint mapping applied to the images, averaged over the whole dataset and expressed
            as coefficients of `basis`.
        """
        mean_b = Volume(
            np.zeros((self.src.L, self.src.L, self.src.L), dtype=self.dtype)
        )

        for i in range(0, self.src.n, self.batch_size):
            im = self.src.images[i : i + self.batch_size]
            batch_mean_b = self.src.im_backward(im, i) / self.src.n
            mean_b += batch_mean_b.astype(self.dtype)

        res = self.basis.evaluate_t(mean_b)
        logger.info(f"Determined adjoint mappings. Shape = {res.shape}")
        return res

    def conj_grad(self, b_coeff, tol=None):
        n = b_coeff.shape[0]
        kernel = self.kernel

        regularizer = config.mean.regularizer
        if regularizer > 0:
            kernel += regularizer

        operator = LinearOperator(
            (n, n), matvec=partial(self.apply_kernel, kernel=kernel), dtype=self.dtype
        )
        if self.precond_kernel is None:
            M = None
        else:
            precond_kernel = self.precond_kernel
            if regularizer > 0:
                precond_kernel += regularizer
            M = LinearOperator(
                (n, n),
                matvec=partial(self.apply_kernel, kernel=precond_kernel),
                dtype=self.dtype,
            )

        tol = tol or config.mean.cg_tol
        target_residual = tol * norm(b_coeff)

        def cb(xk):
            logger.info(
                f"Delta {norm(b_coeff - self.apply_kernel(xk))} (target {target_residual})"
            )

        x, info = scipy.sparse.linalg.cg(
            operator, b_coeff, M=M, callback=cb, tol=tol, atol=0
        )

        if info != 0:
            raise RuntimeError("Unable to converge!")
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
        vol = self.basis.evaluate(vol_coeff)
        # convolve_volume expects a 3-dimensional array
        # so we remove the first dimension of the volume, which is 1
        vol = Volume(kernel.convolve_volume(vol[0]))
        vol = self.basis.evaluate_t(vol)

        return vol
