import logging
import os
from pathlib import Path

from aspire.basis import Coef, FFBBasis3D
from aspire.reconstruction.kernel import FourierKernel

logger = logging.getLogger(__name__)


class Estimator:
    def __init__(
        self,
        src,
        basis=None,
        batch_size=512,
        preconditioner="circulant",
        checkpoint_iterations=10,
        checkpoint_prefix="volume_checkpoint",
        maxiter=50,
        boost=True,
    ):
        """
        An object representing a 2*L-by-2*L-by-2*L array containing the non-centered Fourier transform of the mean
        least-squares estimator kernel.
        Convolving a volume with this kernel is equal to projecting and backproject-ing that volume in each of the
        projection directions (with the appropriate amplitude multipliers and CTFs) and averaging over the whole
        dataset.
        Note that this is a non-centered Fourier transform, so the zero frequency is found at index 1.

        :param src: `ImageSource` to be used for estimation.
        :param basis: 3D Basis to be used during estimation.
        :param batch_size: Optional batch size of images drawn from
            `src` during back projection and kernel estimation steps.
        :param preconditioner: Optional kernel preconditioner (`string`).
            Currently supported options are "circulant" or None.
        :param checkpoint_iterations: Optionally save `cg` estimated
            `basis` coefficients periodically each
            `checkpoint_iterations`.  Setting to `None` disables,
            otherwise checks for positive integer.  Note, when
            `maxiter` is not `None` and `cg` fails to converge a final
            checkpoint will still be written.
        :param checkpoint_prefix: Optional path prefix for `cg`
            checkpoint files.  If the parent directory does not exist,
            creation is attempted.  `_iter{N}` will be appended to the
            prefix.
        :param maxiter: Optional max number of `cg` iterations
            before returning.  This should be used in conjunction with
            `checkpoint_iterations` to prevent excessive disk usage.
            `None` disables.
        :param boost: Option to use `src` symmetry to boost number of images used for mean estimation (Boolean).
            Default of `True` employs symmetry boosting.
        """

        self.src = src
        if basis is None:
            logger.info(f"{self.__class__.__name__} instantiating default basis.")
            basis = FFBBasis3D(src.L, dtype=src.dtype)
        self.basis = basis
        self.dtype = self.src.dtype
        self.batch_size = batch_size
        if not preconditioner or preconditioner.lower() == "none":
            # Resolve None and string nones to None
            preconditioner = None
        elif preconditioner not in ["circulant"]:
            raise ValueError(
                f"Supplied preconditioner {preconditioner} is not supported."
            )
        self.preconditioner = preconditioner
        self.boost = boost

        # dtype configuration
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

        # Checkpoint configuration
        if checkpoint_iterations is not None:
            try:
                checkpoint_iterations = int(checkpoint_iterations)
            except ValueError:
                # Sentinel value to emit a more descriptive message below.
                checkpoint_iterations = -1

            if not checkpoint_iterations > 0:
                raise ValueError(
                    "`checkpoint_iterations` should be a positive integer or `None`."
                )
        self.checkpoint_iterations = checkpoint_iterations

        # Create checkpointing dirs as needed
        if checkpoint_prefix:
            parent = Path(checkpoint_prefix).parent
            if not os.path.exists(parent):
                os.makedirs(parent)
        self.checkpoint_prefix = checkpoint_prefix

        # Maximum iteration configuration
        if maxiter is not None:
            try:
                maxiter = int(maxiter)
            except ValueError:
                # Sentinel value to emit a more descriptive message below.
                maxiter = -1
            if not maxiter > 0:
                raise ValueError("`maxiter` should be a positive integer or `None`.")
        self.maxiter = maxiter

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
                    1.0 / self.kernel.circularize()
                )
            else:
                precond_kernel = self.precond_kernel = None
            return precond_kernel

        else:
            raise AttributeError(name)

    def compute_kernel(self):
        raise NotImplementedError("Subclasses must implement the compute_kernel method")

    def estimate(self, b_coef=None, x0=None, tol=1e-5, regularizer=0):
        """Return an estimate as a Volume instance."""
        if b_coef is None:
            b_coef = self.src_backward()
        est_coef = self.conj_grad(b_coef, x0=x0, tol=tol, regularizer=regularizer)
        est = Coef(self.basis, est_coef).evaluate()

        return est

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

        vol = Coef(self.basis, vol_coef).evaluate()  # returns a Volume
        vol = kernel.convolve_volume(vol)  # returns a Volume
        vol_coef = self.basis.evaluate_t(vol)
        return vol_coef
