import logging
from functools import partial

import numpy as np
from scipy.linalg import norm
from scipy.sparse.linalg import LinearOperator

from aspire import config
from aspire.basis import Coef
from aspire.nufft import anufft
from aspire.numeric import fft
from aspire.numeric.scipy import cg
from aspire.operators import evaluate_src_filters_on_grid
from aspire.reconstruction import Estimator, FourierKernel, FourierKernelMatrix
from aspire.volume import Volume, rotated_grids

logger = logging.getLogger(__name__)


class WeightedVolumesEstimator(Estimator):
    def __init__(self, weights, *args, **kwargs):
        """
        Weighted mean volume estimation.

        This class holds the `FourierKernelMatrix`, stored as a r x r
        matrix of volumes.  The problem being solved here is the
        minimization given by eq. (14) in the paper, rewritten as the
        normal equations in eq. (20) and more compactly in eq. (23).

        Convolution with each of these kernels is equivalent to
        performing a projection/backprojection on a volume, with the
        appropriate amplitude modifiers and CTF, and also a weighting
        term; the r^2 volumes are each of pairwise products between
        the weighting vectors given by the columns of wts.

        Note that this is a non-centered Fourier transform, so the
        zero frequency is found at index 0.

        Formulas and conventions used for Volume estimation are
        described in:

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
                self.precond_kernel = FourierKernelMatrix(
                    1.0 / self.kernel.circularize()
                )
            else:
                if self.preconditioner and (
                    self.preconditioner.lower() not in ("none")
                ):
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
        Compute and return FourierKernelMatrix instance.
        """
        return FourierKernelMatrix(self._compute_kernel())

    def _compute_kernel(self):
        """
        :return: r x r matrix of volumes, shaped (r, r, 2L, 2L, 2L).
        """

        _2L = 2 * self.src.L
        # Note, because we're iteratively summing it is critical we zero this array.
        kernel = np.zeros((self.r, self.r, _2L, _2L, _2L), dtype=self.dtype)

        # Handle symmetry boosting.
        sym_rots = np.eye(3, dtype=self.dtype)[None]
        if self.boost:
            sym_rots = self.src.symmetry_group.matrices

        for i in range(0, self.src.n, self.batch_size):
            _range = np.arange(i, min(self.src.n, i + self.batch_size), dtype=int)
            sq_filters_f = evaluate_src_filters_on_grid(self.src, _range) ** 2
            amplitudes_sq = (self.src.amplitudes[_range] ** 2).astype(
                self.dtype, copy=False
            )

            for k in range(self.r):
                for j in range(k + 1):
                    weights = sq_filters_f * amplitudes_sq

                    if self.src.L % 2 == 0:
                        weights[0, :, :] = 0
                        weights[:, 0, :] = 0

                    weights *= (
                        self.weights[_range, j] * self.weights[_range, k]
                    ).reshape(1, 1, len(_range))

                    weights = np.transpose(weights, (2, 0, 1)).flatten()

                    # Apply boosting.
                    batch_kernel = np.zeros((_2L, _2L, _2L), dtype=self.dtype)
                    for sym_rot in sym_rots:
                        rotations = sym_rot @ self.src.rotations[_range]
                        pts_rot = rotated_grids(self.src.L, rotations)
                        pts_rot = pts_rot.reshape((3, -1))

                        batch_kernel += (
                            1
                            / (self.r * self.src.L**4)
                            * anufft(weights, pts_rot, (_2L, _2L, _2L), real=True)
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

        return kermat_f

    def src_backward(self):
        """
        Apply adjoint mapping to source

        :return: The adjoint mapping applied to the images, averaged over the whole dataset and expressed
            as coefficients of `basis`.
        """
        # Handle symmetry boosting.
        symmetry_group = None
        sym_order = 1
        if self.boost:
            symmetry_group = self.src.symmetry_group
            sym_order = len(symmetry_group.matrices)

        # src_vols_wt_backward
        vol_rhs = Volume(
            np.zeros((self.r, self.src.L, self.src.L, self.src.L), dtype=self.dtype)
        )

        for i in range(0, self.src.n, self.batch_size):
            for k in range(self.r):
                im = self.src.images[i : i + self.batch_size]

                batch_vol_rhs = self.src.im_backward(
                    im,
                    i,
                    self.weights[:, k],
                    symmetry_group=symmetry_group,
                ) / (self.src.n * sym_order)
                vol_rhs[k] += batch_vol_rhs.astype(self.dtype)

        res = np.sqrt(self.src.n * sym_order) * self.basis.evaluate_t(vol_rhs)
        logger.info(f"Determined weighted adjoint mappings. Shape = {res.shape}")

        return res

    def conj_grad(self, b_coef, x0=None, tol=1e-5, regularizer=0):
        count = b_coef.shape[-1]  # b_coef should be (r, basis.count)
        kernel = self.kernel

        if regularizer > 0:
            kernel += regularizer

        operator = LinearOperator(
            (self.r * count, self.r * count),
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
                (self.r * count, self.r * count),
                matvec=partial(self.apply_kernel, kernel=precond_kernel),
                dtype=self.dtype,
            )

        tol = tol or config.mean.cg_tol
        target_residual = tol * norm(b_coef)

        # callback setup
        self.i = 0  # iteration counter

        def cb(xk):
            self.i += 1  # increment iteration count

            logger.info(
                f"[Iter {self.i}]: Delta {norm(b_coef - self.apply_kernel(xk))} (target {target_residual})"
            )

            # Do checkpoint at `checkpoint_iterations`,
            _do_checkpoint = (
                self.checkpoint_iterations is not None
                and (self.i % self.checkpoint_iterations) == 0
            )
            # or the last iteration when `maxiter` provided.
            if self.maxiter:
                _do_checkpoint |= self.i == (self.maxiter - 1)

            # Optional checkpoint
            if _do_checkpoint:
                # Construct checkpoint path
                path = f"{self.checkpoint_prefix}_iter{self.i:04d}.npy"
                # Write out the current solution
                np.save(path, xk)
                logger.info(f"Checkpoint saved to `{path}`")

        x, info = cg(
            operator,
            b_coef.flatten(),
            x0=x0,
            M=M,
            callback=cb,
            rtol=tol,
            atol=0,
            maxiter=self.maxiter,
        )

        if info != 0:
            logger.warning(
                f"Conjugate gradient unable to converge after {info} iterations."
            )

        return x.reshape(self.r, self.basis.count)

    def apply_kernel(self, vol_coef, kernel=None):
        """
        Applies the kernel represented by convolution

        :param vol_coef: The volume to be convolved, stored in the basis coefficients.
        :param kernel: a Kernel object. If None, the kernel for this Estimator is used.
        :return: The result of evaluating `vol_coef` in the given basis, convolving with the kernel given by
            kernel, and backprojecting into the basis.
        """
        if kernel is None:
            kernel = self.kernel

        assert np.size(vol_coef) == self.r * self.basis.count
        if vol_coef.ndim == 1:
            vol_coef = vol_coef.reshape(self.r, self.basis.count)

        vols_out = Volume(
            np.zeros((self.r, self.src.L, self.src.L, self.src.L), dtype=self.dtype)
        )

        vol = Coef(self.basis, vol_coef).evaluate()

        for k in range(self.r):
            for j in range(self.r):
                vols_out[k] = vols_out[k] + kernel.convolve_volume(vol[j], j, k)
                # Note this is where we would add mask_gamma

        vol_coef = self.basis.evaluate_t(vols_out)

        return vol_coef


class MeanEstimator(WeightedVolumesEstimator):
    """
    Special case of weighted mean volume estimate,
    for a single volume.
    """

    def __init__(self, src, **kwargs):
        # Note, Handle boosting by adjusting weights based on symmetric order.
        weights = np.ones((src.n, 1)) / np.sqrt(
            src.n * len(src.symmetry_group.matrices)
        )
        super().__init__(weights, src, **kwargs)

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
        """
        Compute and return `FourierKernel` instance.
        """
        # Note for the r=1 we select and return a single kernel.
        return FourierKernel(self._compute_kernel()[0][0])
