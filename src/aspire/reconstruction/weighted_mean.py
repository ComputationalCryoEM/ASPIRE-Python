import logging

import numpy as np
from scipy.fftpack import fft2

from aspire.image import Image
from aspire.nufft import anufft
from aspire.reconstruction import FourierKernelMat, MeanEstimator
from aspire.utils import vec_to_symmat_iso, vecmat_to_volmat, volmat_to_vecmat
from aspire.utils.fft import mdim_ifftshift
from aspire.utils.matlab_compat import m_flatten, m_reshape
from aspire.volume import Volume, rotated_grids

logger = logging.getLogger(__name__)


class WeightedVolumesEstimator(MeanEstimator):
    def __init__(weights, *args, **kwargs):
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
        assert self.n == self.weights.shape[0]
        self.r = self.weights.shape[1]
        super().__init__(*args, **kwargs)

    def compute_kernel(self):
        """
        :return: r x r matrix of volumes, shaped (r, r, 2L, 2L, 2L).
        """

        _2L = 2 * self.L
        kernel = np.zeros((r, r, _2L, _2L, _2L), dtype=self.dtype)
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

        kermat_f = np.empty((self.r, self.r, self.M))
        logger.info("Computing non-centered Fourier Transform")
        for k in range(self.r):
            for j in range(k):
                kernel = mdim_ifftshift(kernel[k, j], range(0, 3))
                # should this be fft3?
                kernel_f = fft2(kernel[k, j], axes=(0, 1, 2))
                kernel_f = np.real(kernel_f)
                kermat_f[k, j] = kernel_f
                kermat_f[j, k] = kermat_f[k, j]

        return FourierKernelMat(kermat_f, centered=False)

    def src_backward(self, mean_vol, noise_variance, shrink_method=None):
        # src_vols_wt_backward

        mean_b = np.zeros((self.r, self.L, self.L, self.L), dtype=self.dtype)

        for k in range(self.r):
            for i in range(0, self.n, self.batch_size):
                im = self.src.images(i, self.batch_size)

                batch_mean_b = self.src.im_backward(im, i, self.weights) / self.n
                mean_b[k] += batch_mean_b.astype(self.dtype)

        res = np.sqrt(self.n) * self.basis.evaluate_t(mean_b)
        logger.info(f"Determined weighted adjoint mappings. Shape = {res.shape}")

        return res

    # def conj_grad(self, b_coeff, tol=None):
    #     # conj_grad_vols_wt, tbd
    #     pass
