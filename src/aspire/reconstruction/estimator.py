import logging

from aspire.reconstruction.kernel import FourierKernel

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
                    1.0 / self.kernel.circularize()
                )
            else:
                precond_kernel = self.precond_kernel = None
            return precond_kernel

        else:
            raise AttributeError(name)

    def compute_kernel(self):
        raise NotImplementedError("Subclasses must implement the compute_kernel method")

    def estimate(self, b_coeff=None, tol=1e-5, regularizer=0):
        """Return an estimate as a Volume instance."""
        if b_coeff is None:
            b_coeff = self.src_backward()
        est_coeff = self.conj_grad(b_coeff, tol=tol, regularizer=regularizer)
        est = self.basis.evaluate(est_coeff).T

        return est

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
        vol = self.basis.evaluate(vol_coeff)  # returns a Volume
        vol = kernel.convolve_volume(vol)  # returns a Volume
        vol_coef = self.basis.evaluate_t(vol)
        return vol_coef
