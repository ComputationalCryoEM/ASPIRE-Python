import logging

import numpy as np
from scipy.fftpack import fft2

from aspire.nufft import anufft
from aspire.reconstruction import Estimator, FourierKernel
from aspire.utils.fft import mdim_ifftshift
from aspire.utils.matlab_compat import m_flatten, m_reshape
from aspire.volume import rotated_grids

logger = logging.getLogger(__name__)


class MeanEstimator(Estimator):
    def compute_kernel(self):
        _2L = 2 * self.L
        kernel = np.zeros((_2L, _2L, _2L), dtype=self.dtype)
        sq_filters_f = self.src.eval_filter_grid(self.L, power=2)

        for i in range(0, self.n, self.batch_size):
            _range = np.arange(i, min(self.n, i + self.batch_size), dtype=int)
            pts_rot = rotated_grids(self.L, self.src.rots[_range, :, :])
            weights = sq_filters_f[:, :, _range]
            weights *= self.src.amplitudes[_range] ** 2

            if self.L % 2 == 0:
                weights[0, :, :] = 0
                weights[:, 0, :] = 0

            pts_rot = m_reshape(pts_rot, (3, -1))
            weights = m_flatten(weights)

            kernel += (
                1
                / (self.n * self.L ** 4)
                * anufft(weights, pts_rot, (_2L, _2L, _2L), real=True)
            )

        # Ensure symmetric kernel
        kernel[0, :, :] = 0
        kernel[:, 0, :] = 0
        kernel[:, :, 0] = 0

        logger.info("Computing non-centered Fourier Transform")
        kernel = mdim_ifftshift(kernel, range(0, 3))
        kernel_f = fft2(kernel, axes=(0, 1, 2))
        kernel_f = np.real(kernel_f)

        return FourierKernel(kernel_f, centered=False)
