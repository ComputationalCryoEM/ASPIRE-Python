import logging
from functools import partial

import numpy as np
from scipy.linalg import norm
from scipy.sparse.linalg import LinearOperator

from aspire.nufft import anufft
from aspire.numeric import fft
from aspire.numeric.scipy import cg
from aspire.operators import evaluate_src_filters_on_grid
from aspire.reconstruction import Estimator, FourierKernel, MeanEstimator
from aspire.utils import (
    make_symmat,
    symmat_to_vec_iso,
    trange,
    vec_to_symmat_iso,
    vecmat_to_volmat,
    volmat_to_vecmat,
)
from aspire.volume import Volume, rotated_grids

logger = logging.getLogger(__name__)


class CovarianceEstimator(Estimator):
    def __init__(self, *args, **kwargs):
        if "mean_kernel" in kwargs:
            self.mean_kernel = kwargs.pop("mean_kernel")
        super().__init__(*args, **kwargs)

    def __getattr__(self, name):
        """Lazy attributes instantiated on first-access"""

        if name == "mean_kernel":
            mean_kernel = self.mean_kernel = MeanEstimator(self.src, self.basis).kernel
            return mean_kernel
        return super(CovarianceEstimator, self).__getattr__(name)

    def compute_kernel(self):
        # TODO: Most of this stuff is duplicated in MeanEstimator - move up the hierarchy?
        n = self.src.n
        L = self.src.L
        _2L = 2 * self.src.L

        kernel = np.zeros((_2L, _2L, _2L, _2L, _2L, _2L), dtype=self.dtype)
        sq_filters_f = np.square(evaluate_src_filters_on_grid(self.src))

        for i in trange(0, n, self.batch_size):
            _range = np.arange(i, min(n, i + self.batch_size))
            pts_rot = rotated_grids(L, self.src.rotations[_range, :, :])
            weights = sq_filters_f[:, :, _range]
            weights *= self.src.amplitudes[_range] ** 2

            if L % 2 == 0:
                weights[0, :, :] = 0
                weights[:, 0, :] = 0

            # TODO: This is where this differs from MeanEstimator
            pts_rot = np.moveaxis(pts_rot, 1, 0).reshape(-1, 3, L**2)
            weights = weights.T.reshape((-1, L**2))

            batch_n = weights.shape[0]
            factors = np.zeros((batch_n, _2L, _2L, _2L), dtype=self.dtype)

            for j in range(batch_n):
                factors[j] = anufft(weights[j], pts_rot[j], (_2L, _2L, _2L), real=True)

            factors = Volume(factors).to_vec()
            kernel += (factors.T @ factors).reshape((_2L,) * 6) / (n * L**8)

        # Ensure symmetric kernel
        kernel[0, :, :, :, :, :] = 0
        kernel[:, 0, :, :, :, :] = 0
        kernel[:, :, 0, :, :, :] = 0
        kernel[:, :, :, 0, :, :] = 0
        kernel[:, :, :, :, 0, :] = 0
        kernel[:, :, :, :, :, 0] = 0

        logger.info("Computing non-centered Fourier Transform")
        kernel = fft.mdim_ifftshift(kernel, range(0, 6))
        kernel_f = fft.fftn(kernel)
        # Kernel is always symmetric in spatial domain and therefore real in Fourier
        kernel_f = np.real(kernel_f)

        return FourierKernel(kernel_f)

    def estimate(self, mean_vol, noise_variance, tol=1e-5, regularizer=0):
        logger.info("Running Covariance Estimator")
        b_coef = self.src_backward(mean_vol, noise_variance)
        est_coef = self.conj_grad(b_coef, tol=tol, regularizer=regularizer)
        covar_est = self.basis.mat_evaluate(est_coef)
        # Note, notice these cancel out, but requires a lot of changes elsewhere in this file,
        # basically totally removing all the `utils/matrix` hacks ... todo.
        covar_est = vecmat_to_volmat(make_symmat(volmat_to_vecmat(covar_est)))
        return covar_est

    def conj_grad(self, b_coef, tol=1e-5, regularizer=0):
        b_coef = symmat_to_vec_iso(b_coef)
        N = b_coef.shape[0]
        kernel = self.kernel

        if regularizer > 0:
            kernel += regularizer

        operator = LinearOperator(
            (N, N),
            matvec=partial(self.apply_kernel, kernel=kernel, packed=True),
            dtype=self.dtype,
        )
        if self.precond_kernel is None:
            M = None
        else:
            precond_kernel = self.precond_kernel
            if regularizer > 0:
                precond_kernel += regularizer
            M = LinearOperator(
                (N, N),
                matvec=partial(self.apply_kernel, kernel=precond_kernel, packed=True),
                dtype=self.dtype,
            )

        target_residual = tol * norm(b_coef)

        def cb(xk):
            logger.info(
                f"Delta {norm(b_coef - self.apply_kernel(xk, packed=True))} (target {target_residual})"
            )

        x, info = cg(operator, b_coef, M=M, callback=cb, rtol=tol, atol=0)

        if info != 0:
            raise RuntimeError("Unable to converge!")
        return vec_to_symmat_iso(x)

    def apply_kernel(self, coef, kernel=None, packed=False):
        """
        Applies the kernel represented by convolution

        :param coef: The volume matrix (6 dimensions) to be convolved (but see the `packed` argument below).
        :param kernel: a Kernel object. If None, the kernel for this Estimator is used.
        :param packed: whether the `coef` matrix represents an isometrically mapped packed vector,
            through the `symmat_to_vec_iso` function. In this case, the function expands `coef` into a symmetric
            matrix internally, and returns a packed vector in return.
        :return: The result of evaluating `coef` in the given basis, convolving with the kernel given by
            kernel, and backprojecting into the basis. If `packed` is True, then the isometrically mapped packed
            vector is returned instead.
        """
        if kernel is None:
            kernel = self.kernel
        if packed:
            coef = vec_to_symmat_iso(coef)

        result = self.basis.mat_evaluate_t(
            kernel.convolve_volume_matrix(self.basis.mat_evaluate(coef))
        )
        return symmat_to_vec_iso(result) if packed else result

    def src_backward(self, mean_vol, noise_variance, shrink_method=None):
        """
        Apply adjoint mapping to source

        :return: The sum of the outer products of the mean-subtracted images in `src`, corrected by the expected noise
            contribution and expressed as coefficients of `basis`.
        """
        covar_b = np.zeros(
            (self.src.L, self.src.L, self.src.L, self.src.L, self.src.L, self.src.L),
            dtype=self.dtype,
        )

        for i in range(0, self.src.n, self.batch_size):
            im = self.src.images[i : i + self.batch_size]
            batch_n = im.n_images
            im_centered = im - self.src.vol_forward(mean_vol, i, self.batch_size)

            im_centered_b = np.zeros(
                (batch_n, self.src.L, self.src.L, self.src.L), dtype=self.dtype
            )
            for j in range(batch_n):
                im_centered_b[j] = self.src.im_backward(im_centered[j], i + j)
            im_centered_b = Volume(im_centered_b).to_vec()

            covar_b += (im_centered_b.T @ im_centered_b).reshape(
                (self.src.L,) * 6
            ) / self.src.n

        covar_b_coef = self.basis.mat_evaluate_t(covar_b)
        return self._shrink(covar_b_coef, noise_variance, shrink_method)

    def _shrink(self, covar_b_coef, noise_variance, method=None):
        """
        Shrink covariance matrix

        :param covar_b_coef: Outer products of the mean-subtracted images
        :param noise_variance: Noise variance
        :param method: One of None/'frobenius_norm'/'operator_norm'/'soft_threshold'
        :return: Shrunk covariance matrix
        """
        assert method in (
            None,
            "frobenius_norm",
            "operator_norm",
            "soft_threshold",
        ), "Unsupported shrink method"

        An = self.basis.mat_evaluate_t(self.mean_kernel.toeplitz())
        if method is None:
            covar_b_coef -= noise_variance * An
        else:
            raise NotImplementedError("Only default shrink method supported.")

        return covar_b_coef
