import logging
import numpy as np
from functools import partial
from scipy.sparse.linalg import LinearOperator, cg
from tqdm import tqdm
from aspyre.estimation.kernel import FourierKernel

logger = logging.getLogger(__name__)


class Estimator:
    def __init__(self, src, basis, as_type='single', batch_size=512, preconditioner='circulant'):
        self.src = src
        self.basis = basis
        self.as_type = as_type
        self.batch_size = batch_size
        self.preconditioner = preconditioner

        self.L = src.L
        self.n = src.n

        """
        An object representing a 2*L-by-2*L-by-2*L array containing the non-centered Fourier transform of the mean
        least-squares estimator kernel.
        Convolving a volume with this kernel is equal to projecting and backproject-ing that volume in each of the
        projection directions (with the appropriate amplitude multipliers and CTFs) and averaging over the whole
        dataset.
        Note that this is a non-centered Fourier transform, so the zero frequency is found at index 1.
        """

    def __getattr__(self, name):
        """Lazy attributes instantiated on first-access"""

        if name == 'kernel':
            logger.info('Computing kernel')
            kernel = self.kernel = self.compute_kernel()
            return kernel

        elif name == 'precond_kernel':
            if self.preconditioner == 'circulant':
                logger.info('Computing Preconditioner kernel')
                precond_kernel = self.precond_kernel = FourierKernel(1. / self.kernel.circularize(), centered=True)
            else:
                precond_kernel = self.precond_kernel = None
            return precond_kernel

        return super(Estimator, self).__getattr__(name)

    def compute_kernel(self):
        raise NotImplementedError('Subclasses must implement the compute_kernel method')

    def estimate(self, b_coeff=None):
        if b_coeff is None:
            b_coeff = self.src_backward()
        est_coeff = self.conj_grad(b_coeff)
        est = self.basis.evaluate(est_coeff)

        return est

    def src_backward(self):
        """
        Apply adjoint mapping to source

        :return: The adjoint mapping applied to the images, averaged over the whole dataset and expressed
            as coefficients of `basis`.
        """
        mean_b = np.zeros((self.L, self.L, self.L), dtype=self.as_type)

        for i in range(0, self.n, self.batch_size):
            im = self.src.images(i, self.batch_size)
            batch_mean_b = self.src.im_backward(im, i) / self.n
            mean_b += batch_mean_b.astype(self.as_type)

        res = self.basis.evaluate_t(mean_b)
        logger.info(f'Determined adjoint mappings. Shape = {res.shape}')
        return res

    def conj_grad(self, b_coeff):
        n = b_coeff.shape[0]
        operator = LinearOperator((n, n), matvec=self.apply_kernel)
        if self.precond_kernel is None:
            M = None
        else:
            M = LinearOperator((n, n), matvec=partial(self.apply_kernel, kernel=self.precond_kernel))

        pbar = tqdm(desc='Running Conjugate Gradient Optimizer')
        x, info = cg(operator, b_coeff, M=M, callback=lambda xk: pbar.update())
        pbar.close()

        if info != 0:
            raise RuntimeError('Unable to converge!')
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
        vol = self.basis.evaluate(vol_coeff).squeeze()
        vol = kernel.convolve_volume(vol)
        vol = self.basis.evaluate_t(vol)

        return vol


