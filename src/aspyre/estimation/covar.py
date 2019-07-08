import logging
import numpy as np
from scipy.fftpack import fftn
import scipy.sparse.linalg
from scipy.sparse.linalg import LinearOperator
from scipy.linalg import norm
from tqdm import tqdm
from functools import partial

from aspyre import config
from aspyre.imaging.threed import rotated_grids
from aspyre.nfft import anufft3
from aspyre.utils.fft import mdim_ifftshift
from aspyre.utils import ensure
from aspyre.utils.matrix import vol_to_vec, vecmat_to_volmat, volmat_to_vecmat, symmat_to_vec_iso, vec_to_symmat_iso, \
    make_symmat
from aspyre.utils.matlab_compat import m_reshape
from aspyre.estimation import Estimator
from aspyre.estimation.mean import MeanEstimator
from aspyre.estimation.kernel import FourierKernel

logger = logging.getLogger(__name__)


class CovarianceEstimator(Estimator):

    def __init__(self, *args, **kwargs):
        if 'mean_kernel' in kwargs:
            self.mean_kernel = kwargs.pop('mean_kernel')
        super().__init__(*args, **kwargs)

    def __getattr__(self, name):
        """Lazy attributes instantiated on first-access"""

        if name == 'mean_kernel':
            mean_kernel = self.mean_kernel = MeanEstimator(self.src, self.basis).kernel
            return mean_kernel
        return super(CovarianceEstimator, self).__getattr__(name)

    def compute_kernel(self):
        # TODO: Most of this stuff is duplicated in MeanEstimator - move up the hierarchy?
        n = self.n
        L = self.L
        _2L = 2 * self.L

        kernel = np.zeros((_2L, _2L, _2L, _2L, _2L, _2L), dtype=self.as_type)
        filters_f = self.src.filters.evaluate_grid(L)
        sq_filters_f = np.array(filters_f ** 2, dtype=self.as_type)

        for i in tqdm(range(0, n, self.batch_size)):
            pts_rot = rotated_grids(L, self.src.rots[:, :, i:i+self.batch_size])
            weights = sq_filters_f[:, :, self.src.filters.indices[i:i+self.batch_size]]
            weights *= self.src.amplitudes[i:i+self.batch_size] ** 2

            if L % 2 == 0:
                weights[0, :, :] = 0
                weights[:, 0, :] = 0

            # TODO: This is where this differs from MeanEstimator
            pts_rot = m_reshape(pts_rot, (3, L**2, -1))
            weights = m_reshape(weights, (L**2, -1))

            batch_n = weights.shape[-1]
            factors = np.zeros((_2L, _2L, _2L, batch_n), dtype=self.as_type)

            # TODO: Numpy has got to have a functional shortcut to avoid looping like this!
            for j in range(batch_n):
                factors[:, :, :, j] = anufft3(weights[:, j], pts_rot[:, :, j], (_2L, _2L, _2L), real=True)

            factors = vol_to_vec(factors)
            kernel += vecmat_to_volmat(factors @ factors.T) / (n * L**8)

        # Ensure symmetric kernel
        kernel[0, :, :, :, :, :] = 0
        kernel[:, 0, :, :, :, :] = 0
        kernel[:, :, 0, :, :, :] = 0
        kernel[:, :, :, 0, :, :] = 0
        kernel[:, :, :, :, 0, :] = 0
        kernel[:, :, :, :, :, 0] = 0

        logger.info('Computing non-centered Fourier Transform')
        kernel = mdim_ifftshift(kernel, range(0, 6))
        kernel_f = fftn(kernel)
        # Kernel is always symmetric in spatial domain and therefore real in Fourier
        kernel_f = np.real(kernel_f)

        return FourierKernel(kernel_f, centered=False)

    def estimate(self, mean_vol, noise_variance):
        logger.info('Running Covariance Estimator')
        b_coeff = self.src_backward(mean_vol, noise_variance)
        est_coeff = self.conj_grad(b_coeff)
        covar_est = self.basis.mat_evaluate(est_coeff)
        covar_est = vecmat_to_volmat(
            make_symmat(
                volmat_to_vecmat(covar_est)
            )
        )
        return covar_est

    def conj_grad(self, b_coeff):
        # TODO: Support regularizer when solving for volume covariance

        b_coeff = symmat_to_vec_iso(b_coeff)
        N = b_coeff.shape[0]

        operator = LinearOperator((N, N), matvec=partial(self.apply_kernel, kernel=self.kernel, packed=True))
        if self.precond_kernel is None:
            M = None
        else:
            M = LinearOperator((N, N), matvec=partial(self.apply_kernel, kernel=self.precond_kernel, packed=True))

        tol = config.covar.cg_tol
        target_residual = tol * norm(b_coeff)

        def cb(xk):
            logger.info(f'Delta {norm(b_coeff - self.apply_kernel(xk, packed=True))} (target {target_residual})')

        x, info = scipy.sparse.linalg.cg(operator, b_coeff, M=M, callback=cb, tol=tol)

        if info != 0:
            raise RuntimeError('Unable to converge!')
        return vec_to_symmat_iso(x)

    def apply_kernel(self, coeff, kernel=None, packed=False):
        """
        Applies the kernel represented by convolution
        :param coeff: The volume matrix (6 dimensions) to be convolved (but see the `packed` argument below).
        :param kernel: a Kernel object. If None, the kernel for this Estimator is used.
        :param packed: whether the `coeff` matrix represents an isometrically mapped packed vector,
            through the `symmat_to_vec_iso` function. In this case, the function expands `coeff` into a symmetric
            matrix internally, and returns a packed vector in return.
        :return: The result of evaluating `coeff` in the given basis, convolving with the kernel given by
            kernel, and backprojecting into the basis. If `packed` is True, then the isometrically mapped packed
            vector is returned instead.
        """
        if kernel is None:
            kernel = self.kernel
        if packed:
            coeff = vec_to_symmat_iso(coeff)

        result = self.basis.mat_evaluate_t(
            kernel.convolve_volume_matrix(
                self.basis.mat_evaluate(coeff)
            )
        )
        return symmat_to_vec_iso(result) if packed else result

    def src_backward(self, mean_vol, noise_variance, shrink_method=None):
        """
        Apply adjoint mapping to source

        :return: The sum of the outer products of the mean-subtracted images in `src`, corrected by the expected noise
        contribution and expressed as coefficients of `basis`.
        """
        covar_b = np.zeros((self.L, self.L, self.L, self.L, self.L, self.L), dtype=self.as_type)

        for i in range(0, self.n, self.batch_size):
            im = self.src.images(i, self.batch_size)
            batch_n = im.shape[-1]
            im_centered = im - self.src.vol_forward(mean_vol, i, self.batch_size)

            im_centered_b = np.zeros((self.L, self.L, self.L, batch_n), dtype=self.as_type)
            for j in range(batch_n):
                im_centered_b[:, :, :, j] = self.src.im_backward(im_centered[:, :, j], i+j)
            im_centered_b = vol_to_vec(im_centered_b)

            covar_b += vecmat_to_volmat(im_centered_b @ im_centered_b.T) / self.n

        covar_b_coeff = self.basis.mat_evaluate_t(covar_b)
        return self._shrink(covar_b_coeff, noise_variance, shrink_method)

    def _shrink(self, covar_b_coeff, noise_variance, method=None):
        """
        Shrink covariance matrix
        :param covar_b_coeff: Outer products of the mean-subtracted images
        :param noise_variance: Noise variance
        :param method: One of None/'frobenius_norm'/'operator_norm'/'soft_threshold'
        :return: Shrunk covariance matrix
        """
        ensure(method in (None, 'frobenius_norm', 'operator_norm', 'soft_threshold'), 'Unsupported shrink method')

        An = self.basis.mat_evaluate_t(self.mean_kernel.toeplitz())
        if method is None:
            covar_b_coeff -= noise_variance * An
        else:
            raise NotImplementedError('Only default shrink method supported.')

        return covar_b_coeff
