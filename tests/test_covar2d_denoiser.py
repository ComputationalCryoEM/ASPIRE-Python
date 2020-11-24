from unittest import TestCase

import numpy as np

from aspire.basis.ffb_2d import FFBBasis2D
from aspire.denoising.denoiser_cov2d import DenoiserCov2D
from aspire.operators.filters import RadialCTFFilter, ScalarFilter
from aspire.source.simulation import Simulation


class BatchedRotCov2DTestCase(TestCase):
    def testMSE(self):
        # need larger numbers of images and higher resolution for good MSE
        dtype = np.float32
        img_size = 64
        num_imgs = 1024
        noise_var = 0.1848
        noise_filter = ScalarFilter(dim=2, value=noise_var)
        filters = [
            RadialCTFFilter(5, 200, defocus=d, Cs=2.0, alpha=0.1)
            for d in np.linspace(1.5e4, 2.5e4, 7)
        ]
        # set simulation object
        sim = Simulation(
            L=img_size,
            n=num_imgs,
            unique_filters=filters,
            offsets=0.0,
            amplitudes=1.0,
            dtype=dtype,
            noise_filter=noise_filter,
        )
        imgs_clean = sim.projections()

        # Specify the fast FB basis method for expending the 2D images
        ffbbasis = FFBBasis2D((img_size, img_size), dtype=dtype)
        denoiser = DenoiserCov2D(sim, ffbbasis, noise_var)
        denoised_src = denoiser.denoise(batch_size=64)
        imgs_denoised = denoised_src.images(0, num_imgs)
        # Calculate the normalized RMSE of the estimated images.
        nrmse_ims = (imgs_denoised - imgs_clean).norm() / imgs_clean.norm()

        self.assertTrue(nrmse_ims < 0.25)
