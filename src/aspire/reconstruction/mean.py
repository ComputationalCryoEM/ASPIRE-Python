import logging
from functools import partial

import numpy as np
from scipy.linalg import norm
from scipy.sparse.linalg import LinearOperator, cg

from aspire import config
from aspire.nufft import anufft
from aspire.numeric import fft
from aspire.operators import evaluate_src_filters_on_grid
from aspire.reconstruction import Estimator, FourierKernel, FourierKernelMat
from aspire.volume import Volume, rotated_grids

logger = logging.getLogger(__name__)


class WeightedVolumesEstimator(Estimator):
    def __init__(self, weights, *args, **kwargs):
        """
        Weighted mean volume estimation.

        This is best considered as an r x r matrix of volumes;
        each volume is a weighted mean least-squares estimator kernel (MeanEstimator).
        Convolution with each of these kernels is equivalent
        to performing a projection/backprojection on a volume,
        with the appropriate amplitude modifiers and CTF,
        and also a weighting term;
        the r^2 volumes are each of pairwise products between the weighting vectors given by the columns of wts.

        Note that this is a non-centered Fourier transform, so the zero frequency is found at index 0.

        Formulas and conventions used for Volume estimation are described in:

        “Structural Variability from Noisy Tomographic Projections.”
        Andén, Joakim, and Amit Singer.
        SIAM journal on imaging sciences vol. 11,2 (2018): 1441-1492.
        doi:10.1137/17M1153509

        "Cryo-EM reconstruction of continuous heterogeneity by Laplacian spectral volumes"
        Amit Moscovich, Amit Halevi, Joakim Andén and Amit Singer
        Inverse Problems, Volume 36, Number 2, 2020 IOP Publishing Ltd
        Special Issue on Cryo-Electron Microscopy and Inverse Problems
        https://doi.org/10.1088/1361-6420/ab4f55

        :param weights: Matrix of weights, n x r.
        """

        self.weights = weights
        self.r = self.weights.shape[1]
        super().__init__(*args, **kwargs)
        assert self.src.n == self.weights.shape[0]

    def __getattr__(self, name):
        if name == "precond_kernel":
            if self.preconditioner == "circulant":
                # TODO: Discuss precond plans.
                # self.precond_kernel = FourierKernelMat(
                #     1.0 / self.kernel.circularize(), centered=True
                # )
                raise NotImplementedError(
                    "Circulant preconditioner not implemented for WeightedVolumesEstimator."
                )
            else:
                if self.preconditioner.lower() not in (None, "none"):
                    logger.warning(
                        f"Preconditioner {self.preconditioner} is not implemented, resetting to default of None."
                    )
                self.precond_kernel = None
            return self.precond_kernel

        else:
            #  raise AttributeError(name)
            return super().__getattr__(name)

    def compute_kernel(self):
        """
        :return: r x r matrix of volumes, shaped (r, r, 2L, 2L, 2L).
        """

        _2L = 2 * self.src.L
        # Note, because we're iteratively summing it is critical we zero this array.
        kernel = np.zeros((self.r, self.r, _2L, _2L, _2L), dtype=self.dtype)
        sq_filters_f = np.square(evaluate_src_filters_on_grid(self.src))

        for k in range(self.r):
            for j in range(k + 1):
                for i in range(0, self.src.n, self.batch_size):
                    _range = np.arange(
                        i, min(self.src.n, i + self.batch_size), dtype=int
                    )
                    pts_rot = rotated_grids(
                        self.src.L, self.src.rotations[_range, :, :]
                    )
                    weights = sq_filters_f[:, :, _range]
                    weights *= self.src.amplitudes[_range] ** 2

                    if self.src.L % 2 == 0:
                        weights[0, :, :] = 0
                        weights[:, 0, :] = 0

                    weights *= (
                        self.weights[_range, j] * self.weights[_range, k]
                    ).reshape(1, 1, len(_range))

                    pts_rot = pts_rot.reshape((3, -1))
                    weights = np.transpose(weights, (2, 0, 1)).flatten()

                    batch_kernel = (
                        1
                        / (2 * self.src.L**4)
                        * anufft(weights, pts_rot[::-1], (_2L, _2L, _2L), real=True)
                    )
                    kernel[k, j] += batch_kernel

                    # r x r symmetric
                    # accumulate batch entries of kernel[k,j] to kernel[j,k]
                    if j != k:
                        kernel[j, k] += batch_kernel

        kermat_f = np.zeros((self.r, self.r, _2L, _2L, _2L))
        logger.info("Computing non-centered Fourier Transform Kernel Mat")
        for k in range(self.r):
            for j in range(self.r):
                # Ensure symmetric kernel
                kernel[k, j, 0, :, :] = 0
                kernel[k, j, :, 0, :] = 0
                kernel[k, j, :, :, 0] = 0

                kernel[k, j] = fft.mdim_ifftshift(kernel[k, j], range(0, 3))
                kernel_f = fft.fftn(kernel[k, j], axes=(0, 1, 2))

                kernel_f = np.real(kernel_f)
                kermat_f[k, j] = kernel_f

        return FourierKernelMat(kermat_f, centered=False)

    def src_backward(self):
        """
        Apply adjoint mapping to source

        :return: The adjoint mapping applied to the images, averaged over the whole dataset and expressed
            as coefficients of `basis`.
        """

        # src_vols_wt_backward
        mean_b = Volume(
            np.zeros((self.r, self.src.L, self.src.L, self.src.L), dtype=self.dtype)
        )

        for k in range(self.r):
            for i in range(0, self.src.n, self.batch_size):
                im = self.src.images[i : i + self.batch_size]

                batch_mean_b = (
                    self.src.im_backward(im, i, self.weights[:, k]) / self.src.n
                )
                mean_b[k] += batch_mean_b.astype(self.dtype)

        res = np.sqrt(self.src.n) * self.basis.evaluate_t(mean_b)
        logger.info(f"Determined weighted adjoint mappings. Shape = {res.shape}")

        return res

    def conj_grad(self, b_coeff, tol=1e-5, regularizer=0):
        n = b_coeff.shape[-1]  # 0???
        kernel = self.kernel

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

        x, info = cg(operator, b_coeff.flatten(), M=M, callback=cb, tol=tol, atol=0)

        if info != 0:
            raise RuntimeError("Unable to converge!")

        # Thinking might be clearer if r x ... but would need to mess with roll/unroll in FBB.
        # return x.reshape(self.r, -1)
        return x.reshape(self.r, self.basis.count)

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

        vols_out = Volume(
            np.zeros((self.r, self.src.L, self.src.L, self.src.L), dtype=self.dtype)
        )

        vol = self.basis.evaluate(vol_coeff)

        for k in range(self.r):
            for j in range(self.r):
                vols_out[k] = vols_out[k] + kernel.convolve_volume(vol[j], j, k)
                # Note this is where we would add mask_gamma

        vol_coeff = self.basis.evaluate_t(vols_out)

        return vol_coeff


class MeanEstimator(WeightedVolumesEstimator):
    """
    Special case of weighted mean volume estimate,
    for a single volume.
    """

    def __init__(self, src, basis, **kwargs):
        weights = np.ones((src.n, 1)) / np.sqrt(src.n)
        super().__init__(weights, src, basis, **kwargs)

    def __getattr__(self, name):
        """
        See `Estimator.__getattr__`.
        """
        return super(WeightedVolumesEstimator, self).__getattr__(name)

    def apply_kernel(self, *args, **kwargs):
        """
        See `Estimator.apply_kernel`.
        """
        return super(WeightedVolumesEstimator, self).apply_kernel(*args, **kwargs)

    def compute_kernel(self):
        _2L = 2 * self.src.L
        kernel = np.zeros((_2L, _2L, _2L), dtype=self.dtype)
        sq_filters_f = np.square(evaluate_src_filters_on_grid(self.src))

        for i in range(0, self.src.n, self.batch_size):
            _range = np.arange(i, min(self.src.n, i + self.batch_size), dtype=int)
            pts_rot = rotated_grids(self.src.L, self.src.rotations[_range, :, :])
            weights = sq_filters_f[:, :, _range]
            weights *= self.src.amplitudes[_range] ** 2

            if self.src.L % 2 == 0:
                weights[0, :, :] = 0
                weights[:, 0, :] = 0

            pts_rot = pts_rot.reshape((3, -1))
            weights = np.transpose(weights, (2, 0, 1)).flatten()

            kernel += (
                1
                / (self.src.n * self.src.L**4)
                * anufft(weights, pts_rot[::-1], (_2L, _2L, _2L), real=True)
            )

        # Ensure symmetric kernel
        kernel[0, :, :] = 0
        kernel[:, 0, :] = 0
        kernel[:, :, 0] = 0

        logger.info("Computing non-centered Fourier Transform")
        kernel = fft.mdim_ifftshift(kernel, range(0, 3))
        kernel_f = fft.fftn(kernel, axes=(0, 1, 2))
        kernel_f = np.real(kernel_f)

        return FourierKernel(kernel_f, centered=False)
