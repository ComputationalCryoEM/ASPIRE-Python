import logging
import numpy as np
from scipy.fftpack import fft2

from aspyre.imaging.threed import rotated_grids
from aspyre.nfft import anufft3
from aspyre.utils.fft import mdim_ifftshift
from aspyre.utils.matlab_compat import m_reshape, m_flatten
from aspyre.estimation import Estimator
from aspyre.estimation.kernel import FourierKernel

logger = logging.getLogger(__name__)


class MeanEstimator(Estimator):

    def compute_kernel(self):
        _2L = 2 * self.L
        kernel = np.zeros((_2L, _2L, _2L), dtype=self.as_type)
        filters_f = self.src.filters.evaluate_grid(self.L)
        sq_filters_f = np.array(filters_f ** 2, dtype=self.as_type)

        for i in range(0, self.n, self.batch_size):
            pts_rot = rotated_grids(self.L, self.src.rots[:, :, i:i+self.batch_size])
            weights = sq_filters_f[:, :, self.src.filters.indices[i:i+self.batch_size]]
            weights *= self.src.amplitudes[i:i+self.batch_size] ** 2

            if self.L % 2 == 0:
                weights[0, :, :] = 0
                weights[:, 0, :] = 0

            pts_rot = m_reshape(pts_rot, (3, -1))
            weights = m_flatten(weights)

            kernel += 1 / (self.n * self.L ** 4) * anufft3(weights, pts_rot, (_2L, _2L, _2L), real=True)

        # Ensure symmetric kernel
        kernel[0, :, :] = 0
        kernel[:, 0, :] = 0
        kernel[:, :, 0] = 0

        logger.info('Computing non-centered Fourier Transform')
        kernel = mdim_ifftshift(kernel, range(0, 3))
        kernel_f = fft2(kernel, axes=(0, 1, 2))
        kernel_f = np.real(kernel_f)

        return FourierKernel(kernel_f, centered=False)


